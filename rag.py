# rag.py
import os
import re
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
import httpx
from bs4 import BeautifulSoup
from readability import Document
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

BASE_DIR = Path("data")
BASE_DIR.mkdir(exist_ok=True)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBEDDER_NAME = os.getenv("EMBEDDER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDER_NAME)
    return _embedder

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = _clean_text(text)
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+size]
        chunks.append(chunk)
        i += size - overlap
    return [c for c in chunks if c.strip()]

def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_csv(path: Path, max_rows: int = 2000) -> str:
    df = pd.read_csv(path).head(max_rows)
    return df.to_csv(index=False)

def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def fetch_url(url: str, timeout: int = 15) -> str:
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        r = client.get(url)
        r.raise_for_status()
        html = r.text
    doc = Document(html)
    summary_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(summary_html, "html.parser")
    text = soup.get_text(" ")
    return _clean_text(text)

def _user_paths(user_id: int) -> Tuple[Path, Path]:
    udir = BASE_DIR / str(user_id)
    udir.mkdir(parents=True, exist_ok=True)
    index_path = udir / "faiss.index"
    meta_path = udir / "meta.jsonl"
    return udir, index_path, meta_path

def _embed(texts: List[str]) -> np.ndarray:
    emb = _get_embedder()
    vecs = emb.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype("float32")

def _load_index(dim: int, index_path: Path) -> faiss.Index:
    if index_path.exists():
        return faiss.read_index(str(index_path))
    return faiss.IndexFlatIP(dim)

def _save_index(index: faiss.Index, index_path: Path) -> None:
    faiss.write_index(index, str(index_path))

def add_texts(user_id: int, texts: List[str], source: str) -> int:
    if not texts:
        return 0
    chunks = []
    for t in texts:
        chunks.extend(chunk_text(t))
    if not chunks:
        return 0
    vecs = _embed(chunks)
    dim = vecs.shape[1]
    udir, index_path, meta_path = None, None, None
    udir, index_path, meta_path = _user_paths(user_id) + (None,)
    # (we created a helper above, adjust)
    udir = BASE_DIR / str(user_id)
    index_path = udir / "faiss.index"
    meta_path = udir / "meta.jsonl"

    index = _load_index(dim, index_path)
    start = index.ntotal
    index.add(vecs)
    _save_index(index, index_path)

    with open(meta_path, "a", encoding="utf-8") as f:
        for ch in chunks:
            meta = {"source": source, "chunk": ch[:2400], "ts": int(time.time())}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return index.ntotal - start

def search(user_id: int, query: str, k: int = 4) -> List[Dict]:
    udir, index_path, meta_path = _user_paths(user_id)
    if not index_path.exists() or not meta_path.exists():
        return []
    index = faiss.read_index(str(index_path))
    qvec = _embed([query])
    D, I = index.search(qvec, k)
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_lines = [json.loads(line) for line in f]
    for idx in I[0]:
        if 0 <= idx < len(meta_lines):
            metas.append(meta_lines[idx])
    return metas

def ingest_file(user_id: int, file_path: Path) -> int:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        text = load_pdf(file_path)
    elif ext == ".md":
        text = load_md(file_path)
    elif ext == ".csv":
        text = load_csv(file_path)
    elif ext == ".txt":
        text = load_txt(file_path)
    else:
        raise ValueError("Unsupported file type")
    return add_texts(user_id, [text], source=file_path.name)

def ingest_url(user_id: int, url: str) -> int:
    text = fetch_url(url)
    return add_texts(user_id, [text], source=url)

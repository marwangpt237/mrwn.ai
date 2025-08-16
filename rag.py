import os
import faiss
import pickle
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch

# ========== Embeddings ==========
class Embeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts: List[str]):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = self.model(**tokens).last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()


# ========== RAG Store ==========
class RAGStore:
    def __init__(self, index_path="faiss.index", store_path="store.pkl"):
        self.index_path = index_path
        self.store_path = store_path
        self.embeddings = Embeddings()
        self.texts = []
        self.index = None

        if os.path.exists(index_path) and os.path.exists(store_path):
            self.load()

    def add(self, texts: List[str]):
        vectors = self.embeddings.encode(texts)
        if self.index is None:
            self.index = faiss.IndexFlatL2(vectors.shape[1])
        self.index.add(vectors)
        self.texts.extend(texts)
        self.save()

    def search(self, query: str, top_k=3):
        if self.index is None or len(self.texts) == 0:
            return []
        q_vec = self.embeddings.encode([query])
        D, I = self.index.search(q_vec, top_k)
        return [self.texts[i] for i in I[0]]

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.store_path, "rb") as f:
            self.texts = pickle.load(f)

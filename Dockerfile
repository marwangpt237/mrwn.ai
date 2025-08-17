FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# تحديث apt وتثبيت dependencies الأساسية
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    poppler-utils \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# تحديث pip قبل التثبيت
RUN python -m pip install --upgrade pip setuptools wheel

# تعيين مجلد العمل
WORKDIR /app

# نسخ requirements وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ التطبيق
COPY main.py .

# expose port
EXPOSE 8080

# تشغيل التطبيق
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

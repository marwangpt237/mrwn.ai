# استخدم base image صغيرة
FROM python:3.10-slim

# منع الـ pip cache عشان ما يكبر حجم الصورة
ENV PIP_NO_CACHE_DIR=1

# تثبيت المتطلبات الأساسية فقط
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# إنشاء مجلد للعمل
WORKDIR /app

# نسخ الملفات
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway يتطلب PORT
ENV PORT=8080
EXPOSE 8080

# شغل التطبيق
CMD ["python", "main.py"]

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 🔹 System deps (OpenCV + OCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# 🔹 Set EasyOCR model dir explicitly
ENV EASY_OCR_DIR=/models
RUN mkdir -p /models

# 🔹 Pre-download EasyOCR models (CPU)
RUN python - <<'PY'
import easyocr
easyocr.Reader(['en'], gpu=False)
print("EasyOCR models downloaded")
PY

COPY . .

ENV YOLO_MODEL_PATH=license_plate_detector.pt
ENV OCR_LANGS=en
ENV MIN_OCR_CONF=0.4

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app", "--workers", "2", "--threads", "4", "--timeout", "120"]
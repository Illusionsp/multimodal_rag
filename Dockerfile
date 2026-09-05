# ReportIQ - production container
FROM python:3.11-slim

# System deps: poppler (pdf2image + unstructured PDF backend),
# tesseract (OCR), libgl1 (opencv, used by chart-type heuristics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data (uploads, extracted images, chroma_db) - mount a volume here
VOLUME ["/app/data"]

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

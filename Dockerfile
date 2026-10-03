FROM python:3.12-slim

# Tesseract OCR (scanned-PDF fallback) and Poppler (pdf2image's page
# rasterizer) are native binaries this project needs at the OS level -
# they can't come from pip. This is exactly the kind of dependency a
# "works on my machine" venv setup hides until you try to deploy it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true \
    PYTHONUNBUFFERED=1

EXPOSE 8501

# Overridden by docker-compose.yml for the api service (uvicorn api:app ...).
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

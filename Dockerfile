FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
COPY . .
RUN mkdir -p data/raw data/processed models logs 
RUN python src/run_pipeline.py
EXPOSE 8005
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1
CMD ["python", "start_api.py"]
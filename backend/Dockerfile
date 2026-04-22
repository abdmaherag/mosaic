FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first to avoid pulling CUDA (~2GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

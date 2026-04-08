FROM python:3.11-slim

WORKDIR /app

# System deps (safe for pandas/sklearn)
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache layer)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Generate datasets at build time (critical for HF Spaces)
RUN python -m data.generators.dataset_loader
RUN python -m data.generators.corruption

# Hugging Face expects this port
EXPOSE 7860

# Prevent Python buffering (better logs)
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
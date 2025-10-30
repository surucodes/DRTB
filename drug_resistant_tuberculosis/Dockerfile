FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8502

# Install build dependencies needed by some ML packages (xgboost, lightgbm, catboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libomp-dev \
    default-libgomp-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install first (cached layer)
COPY drug_resistant_tuberculosis/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application
COPY . /app

# Create output directories expected by the app
RUN mkdir -p /app/drug_resistant_tuberculosis/outputs /app/drug_resistant_tuberculosis/uploads

EXPOSE 8502

# Use gunicorn to serve the Flask app. Module: package.application:app
CMD gunicorn --bind 0.0.0.0:${PORT:-8502} drug_resistant_tuberculosis.application:app --workers 1 --threads 4

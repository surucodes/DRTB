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
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install first (cached layer)
COPY drug_resistant_tuberculosis/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application
# Copy the rest of the application
# Also copy any pretrained models placed under drug_resistant_tuberculosis/models into the image
COPY . /app

# If there are pretrained models saved under outputs/models include them in the image so the app can
# prefer pretrained models at runtime. The .dockerignore file must allow the path.
COPY drug_resistant_tuberculosis/outputs/models /app/drug_resistant_tuberculosis/outputs/models

# Create output directories expected by the app
RUN mkdir -p /app/drug_resistant_tuberculosis/outputs /app/drug_resistant_tuberculosis/uploads /app/drug_resistant_tuberculosis/models

EXPOSE 8502

# Use gunicorn to serve the Flask app. Module: package.application:app
"""
Use a shell wrapper so environment variable expansion works for ${PORT} at container start.
We keep the JSON exec form but run a shell to allow ${PORT:-8502} substitution.
"""
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8502} drug_resistant_tuberculosis.application:app --workers 1 --threads 4"]

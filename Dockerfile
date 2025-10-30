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

# Copy an entrypoint script that handles PORT expansion and execs the process
COPY drug_resistant_tuberculosis/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint for robust startup and signal handling
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command (entrypoint will run this if no args provided)
CMD ["gunicorn", "--bind", "0.0.0.0:8502", "drug_resistant_tuberculosis.application:app", "--workers", "1", "--threads", "4"]

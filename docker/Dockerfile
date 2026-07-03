# Dockerfile for Cloud-Edge Bayesian Optimization
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config
COPY src/ ./src/
COPY aws_config.json ./
COPY results/ ./results/
COPY paper/ ./paper/

# Create directories for data
RUN mkdir -p /app/data /app/results

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "src/cloud_edge_working.py"]

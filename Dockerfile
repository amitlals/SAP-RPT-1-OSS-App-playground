# Dockerfile for SAP Finance Dashboard with RPT-1-OSS Model
# Multi-stage build to optimize image size for HuggingFace Spaces

# Stage 1: Build wheels for heavy dependencies
FROM python:3.11-slim as builder

WORKDIR /wheels

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create wheels directory for pip to use
RUN mkdir -p /wheels

# Build torch and other ML libraries as wheels (lighter than full install)
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
    torch==2.0.0 \
    torchvision==0.15.0 \
    torchaudio==2.0.0 \
    transformers==4.30.0 2>&1 || true

# Build other dependencies as wheels
RUN pip wheel --no-cache-dir --wheel-dir=/wheels \
    git+https://github.com/SAP-samples/sap-rpt-1-oss 2>&1 || true

# Stage 2: Runtime image (minimal size)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV TORCH_HOME=/app/torch_cache

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install stable gradio version
RUN pip install --no-cache-dir "gradio==4.44.1"

# Copy pre-built wheels from builder stage and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels \
    torch==2.0.0 \
    transformers==4.30.0 2>&1 || true && \
    rm -rf /wheels

# Try installing sap-rpt-oss directly (will use pre-built wheels if available)
RUN pip install --no-cache-dir git+https://github.com/SAP-samples/sap-rpt-1-oss 2>&1 || true

# Copy application code
COPY . .

# Create data directory and torch cache directory
RUN mkdir -p /app/data /app/torch_cache

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app_gradio.py"]

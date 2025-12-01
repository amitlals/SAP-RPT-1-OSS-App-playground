# Dockerfile for SAP Finance Dashboard with RPT-1-OSS Model
# Optimized single-stage build for HuggingFace Spaces

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV TORCH_HOME=/app/torch_cache
ENV HUGGINGFACE_HUB_CACHE=/app/hf_cache

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install base dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install Gradio (pinned for stability)
RUN pip install --no-cache-dir "gradio==4.44.1"

# Install core ML libraries (pre-built wheels, no compilation needed)
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    transformers==4.30.0 \
    scikit-learn==1.2.0

# Install SAP-RPT-1-OSS from GitHub
# Note: Requires HF_TOKEN environment variable for gated model access
RUN pip install --no-cache-dir \
    git+https://github.com/SAP-samples/sap-rpt-1-oss

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/data /app/torch_cache /app/hf_cache

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app_gradio.py"]

# Dockerfile for SAP Finance Dashboard with RPT-1-OSS Model
# Optimized for Azure Container Apps deployment

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies (base packages only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install stable gradio version that avoids JSON schema regression
RUN pip install --no-cache-dir "gradio==4.44.1"

# Install SAP-RPT-1-OSS dependencies and package
RUN pip install --no-cache-dir \
    torch==2.0.0 \
    transformers==4.30.0 \
    scikit-learn==1.2.0 \
    xgboost==1.7.0 \
    lightgbm==3.3.5 \
    catboost==1.2.0 && \
    pip install --no-cache-dir git+https://github.com/SAP-samples/sap-rpt-1-oss || echo "SAP-RPT-1-OSS installation warning"

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p /app/data

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app_gradio.py"]

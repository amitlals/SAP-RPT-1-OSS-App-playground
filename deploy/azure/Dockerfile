# Dockerfile for SAP RPT-1-OSS on Azure Container Apps
# Runs both FastAPI (8000) and Streamlit (8501)

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TABPFN_TOKEN=""

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    streamlit \
    tabpfn-client \
    pandas \
    numpy \
    scikit-learn \
    requests \
    pydantic

# Copy application files
COPY sap_rpt1_api.py .
COPY sap_rpt1_frontend.py .
COPY data/ ./data/

# Create supervisor config
RUN mkdir -p /var/log/supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Create startup script
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with supervisor
CMD ["/app/start.sh"]

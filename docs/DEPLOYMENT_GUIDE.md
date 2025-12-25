# 🚀 Deployment Guide

**Complete guide to deploying SAP RPT-1-OSS applications**

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [HuggingFace Spaces](#huggingface-spaces)
4. [Azure Container Apps](#azure-container-apps)
5. [Docker Standalone](#docker-standalone)
6. [CI/CD Pipeline](#cicd-pipeline)

---

## Prerequisites

### Required

- **Python 3.10+**
- **TabPFN token** from [tabpfn.com](https://tabpfn.com)
- **Git** for version control

### Optional

- **Docker** for containerized deployment
- **Azure CLI** for Azure deployment
- **HuggingFace CLI** for Space management

---

## Local Development

### Step 1: Clone Repository

```bash
git clone https://github.com/amitlals/SAP-RPT-1-OSS-App-playground.git
cd SAP-RPT-1-OSS-App-playground
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your token
TABPFN_ACCESS_TOKEN=your_token_here
```

### Step 5: Run Application

```bash
# Finance Dashboard
streamlit run apps/01-finance-dashboard/app.py

# Forecast Showdown
streamlit run apps/02-forecast-showdown/app.py

# Predictive Integrity
streamlit run apps/03-predictive-integrity/app.py
```

---

## HuggingFace Spaces

### Step 1: Create Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Enter space name
3. Select **Docker** SDK
4. Choose visibility

### Step 2: Upload Files

Upload from the appropriate app folder:

```
apps/01-finance-dashboard/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

### Step 3: Configure Secrets

Go to **Settings → Repository secrets**:

| Secret | Value |
|--------|-------|
| `TABPFN_ACCESS_TOKEN` | Your TabPFN token |
| `SAP_RPT1_TOKEN` | (Optional) SAP-RPT-1 closed API token |

### Step 4: Wait for Build

The space will automatically build and deploy. Check the **Logs** tab for progress.

### Live Spaces

| App | URL |
|-----|-----|
| Finance Dashboard | [sap-finance-dashboard-RPT-1-OSS](https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS) |
| Forecast Showdown | [sap-rpt1-forecast-showdown](https://huggingface.co/spaces/amitgpt/sap-rpt1-forecast-showdown) |
| Predictive Integrity | [sap-predictive-integrity-using-RPT-1](https://huggingface.co/spaces/amitgpt/sap-predictive-integrity-using-RPT-1) |

---

## Azure Container Apps

### Step 1: Prerequisites

```bash
# Install Azure CLI
winget install Microsoft.AzureCLI

# Login to Azure
az login
```

### Step 2: Run Deployment Script

```powershell
cd deploy/azure

# Set environment variables
$env:TABPFN_ACCESS_TOKEN = "your_token_here"

# Run deployment
./deploy-azure.ps1
```

### Step 3: Verify Deployment

```bash
# Get app URL
az containerapp show \
  --name sap-rpt1-oss-app \
  --resource-group rg-sap-finance-dashboard \
  --query "properties.configuration.ingress.fqdn"
```

### Live Deployment

🔗 [sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io](https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io/)

---

## Docker Standalone

### Build Image

```bash
# From app folder
cd apps/01-finance-dashboard
docker build -t sap-finance-dashboard .
```

### Run Container

```bash
docker run -d \
  -p 7860:7860 \
  -e TABPFN_ACCESS_TOKEN=your_token \
  sap-finance-dashboard
```

### Docker Compose

```yaml
version: '3.8'
services:
  finance-dashboard:
    build: ./apps/01-finance-dashboard
    ports:
      - "7860:7860"
    environment:
      - TABPFN_ACCESS_TOKEN=${TABPFN_ACCESS_TOKEN}
    restart: unless-stopped
```

---

## CI/CD Pipeline

### GitHub Actions

The repository includes a workflow at `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Azure

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}
      
      - name: Build and Deploy
        run: |
          az acr build --registry $ACR_NAME --image sap-app:${{ github.sha }} .
          az containerapp update --name $APP_NAME --image $ACR_NAME.azurecr.io/sap-app:${{ github.sha }}
```

### Required Secrets

| Secret | Description |
|--------|-------------|
| `AZURE_CREDENTIALS` | Azure service principal JSON |
| `TABPFN_ACCESS_TOKEN` | TabPFN API token |

---

## 🔒 Security Checklist

- [ ] Store tokens in environment variables or secrets manager
- [ ] Never commit `.env` files
- [ ] Enable HTTPS (auto-enabled on HF/Azure)
- [ ] Rotate tokens periodically
- [ ] Review access logs regularly

---

## 📊 Monitoring

### HuggingFace

- View logs in **Logs** tab
- Check **Settings → Usage** for metrics

### Azure

```bash
# View logs
az containerapp logs show --name sap-rpt1-oss-app --resource-group rg-sap-finance-dashboard

# View metrics
az monitor metrics list --resource $RESOURCE_ID
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| "Token not found" | Set `TABPFN_ACCESS_TOKEN` environment variable |
| Build fails | Check `requirements.txt` for missing packages |
| Port not accessible | Ensure port 7860 is exposed |
| Slow startup | First startup downloads model weights |

### Get Help

- 📖 [TabPFN Docs](https://docs.tabpfn.com)
- 🤗 [HuggingFace Forums](https://discuss.huggingface.co)
- ☁️ [Azure Support](https://azure.microsoft.com/support)

---

## 📚 Related Documentation

- [Architecture Overview](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [Azure Deployment](../deploy/azure/README.md)
- [HuggingFace Deployment](../deploy/huggingface/README.md)

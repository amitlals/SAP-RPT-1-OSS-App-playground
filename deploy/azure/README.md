# ☁️ Azure Container Apps Deployment

**Deploy SAP RPT-1-OSS applications to Azure Container Apps**

---

## 🎯 Overview

This folder contains everything needed to deploy the apps to Azure Container Apps with:
- **Frontend**: Streamlit on port 8501
- **Backend**: FastAPI on port 8000
- **Process Manager**: Supervisord for multi-process management

---

## 📁 Files

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-service container image |
| `deploy-azure.ps1` | PowerShell deployment script |
| `supervisord.conf` | Process manager configuration |
| `start.sh` | Container startup script |

---

## 🚀 Quick Deploy

### Prerequisites

1. **Azure CLI** installed and logged in
2. **Docker** (optional, for local testing)
3. **TabPFN token** from [tabpfn.com](https://tabpfn.com)

### Deploy

```powershell
# Set your TabPFN token
$env:TABPFN_ACCESS_TOKEN = "your_token_here"

# Run deployment script
./deploy-azure.ps1
```

The script will:
1. Create a resource group
2. Create Azure Container Registry
3. Build and push Docker image
4. Create Container App Environment
5. Deploy the application

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TABPFN_ACCESS_TOKEN` | TabPFN API token | Yes |
| `API_URL` | Backend API URL (auto-configured) | No |

### Modify Deployment

Edit `deploy-azure.ps1` to customize:

```powershell
$RESOURCE_GROUP = "rg-sap-finance-dashboard"  # Resource group name
$LOCATION = "eastus2"                          # Azure region
$APP_NAME = "sap-rpt1-oss-app"                # App name
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Azure Container Apps                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │   Streamlit UI  │───▶│       FastAPI Backend           │ │
│  │   (Port 8501)   │    │       (Port 8000)               │ │
│  └─────────────────┘    └───────────────┬─────────────────┘ │
│                                         │                    │
│                                         ▼                    │
│                              ┌─────────────────────┐        │
│                              │   TabPFN Cloud API  │        │
│                              └─────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Cost Estimation

| Resource | Tier | Est. Monthly Cost |
|----------|------|-------------------|
| Container Apps | Consumption | $10-30 |
| Container Registry | Basic | $5 |
| **Total** | | **~$15-35/month** |

---

## 🔒 Security Best Practices

1. **Use Managed Identity** for Azure resources
2. **Store tokens in Key Vault** for production
3. **Enable HTTPS** (auto-enabled by Container Apps)
4. **Configure IP restrictions** if needed

---

## 🐛 Troubleshooting

### View Logs

```bash
az containerapp logs show --name sap-rpt1-oss-app --resource-group rg-sap-finance-dashboard
```

### Restart App

```bash
az containerapp revision restart --name sap-rpt1-oss-app --resource-group rg-sap-finance-dashboard
```

### Check Status

```bash
az containerapp show --name sap-rpt1-oss-app --resource-group rg-sap-finance-dashboard --query "properties.latestRevisionFqdn"
```

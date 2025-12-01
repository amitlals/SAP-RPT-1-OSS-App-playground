# SAP Finance Dashboard - Azure Container Apps Deployment Guide

## Prerequisites

1. **Azure CLI** - Install from: https://aka.ms/installazurecliwindows
2. **Azure Subscription** - Active Azure account
3. **Git** (optional) - For code updates

## Quick Deploy Steps

### Option 1: Automated Deployment Script

Open PowerShell as Administrator and run:

```powershell
cd "c:\Users\amlal\Downloads\VSCode-SAP-AI-Copilot-Projects2025\SAP-RPT-1-OSS-App"
.\deploy-azure.ps1
```

The script will:
- ✅ Create Azure Resource Group
- ✅ Create Azure Container Registry
- ✅ Build Docker image in Azure (no local Docker needed)
- ✅ Create Container Apps Environment
- ✅ Deploy your application
- ✅ Provide public URL

**Deployment Time:** ~10-15 minutes

### Option 2: Manual Deployment (Step-by-Step)

If the automated script doesn't work, follow these manual commands:

#### 1. Login to Azure
```bash
az login
```

#### 2. Set Variables
```bash
set RESOURCE_GROUP=rg-sap-finance-dashboard
set LOCATION=eastus
set ACR_NAME=acrsapfinance%RANDOM%
set ENVIRONMENT_NAME=env-sap-finance
set APP_NAME=sap-finance-dashboard
```

#### 3. Create Resource Group
```bash
az group create --name %RESOURCE_GROUP% --location %LOCATION%
```

#### 4. Create Container Registry
```bash
az acr create --resource-group %RESOURCE_GROUP% --name %ACR_NAME% --sku Basic --admin-enabled true
```

#### 5. Build Docker Image (Using Azure - No local Docker needed)
```bash
az acr build --registry %ACR_NAME% --image sap-finance-dashboard:latest .
```

#### 6. Create Container Apps Environment
```bash
az containerapp env create --name %ENVIRONMENT_NAME% --resource-group %RESOURCE_GROUP% --location %LOCATION%
```

#### 7. Deploy Container App
```bash
az containerapp create ^
  --name %APP_NAME% ^
  --resource-group %RESOURCE_GROUP% ^
  --environment %ENVIRONMENT_NAME% ^
  --image %ACR_NAME%.azurecr.io/sap-finance-dashboard:latest ^
  --registry-server %ACR_NAME%.azurecr.io ^
  --target-port 7862 ^
  --ingress external ^
  --cpu 2 ^
  --memory 4Gi ^
  --min-replicas 1 ^
  --max-replicas 3
```

#### 8. Get Application URL
```bash
az containerapp show --name %APP_NAME% --resource-group %RESOURCE_GROUP% --query properties.configuration.ingress.fqdn -o tsv
```

## Post-Deployment Configuration

### Add Hugging Face Token (Required for AI Features)

1. Go to Azure Portal: https://portal.azure.com
2. Navigate to your Container App
3. Go to **Settings** > **Secrets**
4. Add secret:
   - Name: `huggingface-token`
   - Value: Your Hugging Face token (get from https://huggingface.co/settings/tokens)
5. Go to **Settings** > **Containers** > **Environment variables**
6. Add environment variable:
   - Name: `HUGGINGFACE_TOKEN`
   - Source: Reference a secret
   - Value: Select `huggingface-token`
7. Click **Save** and wait for app to restart

### Optional: Add SAP OData Credentials

If connecting to SAP systems:
- `SAP_ODATA_BASE_URL`: Your SAP OData endpoint
- `SAP_USERNAME`: SAP username (as secret)
- `SAP_PASSWORD`: SAP password (as secret)

## Application Architecture

```
┌─────────────────────────────────────────┐
│     Azure Container Apps                │
│  ┌───────────────────────────────────┐  │
│  │  SAP Finance Dashboard            │  │
│  │  - Gradio Web Interface           │  │
│  │  - RPT-1-OSS AI Model             │  │
│  │  - Port 7862                      │  │
│  └───────────────────────────────────┘  │
│              ↕                          │
│  ┌───────────────────────────────────┐  │
│  │  Azure Container Registry         │  │
│  │  - Docker Image Storage           │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Application Features

Once deployed, you'll have access to:

1. **📊 Dashboard** - Financial metrics and visualizations
2. **🔍 Data Explorer** - Browse datasets
3. **📤 Upload** - Upload custom CSV files
4. **🤖 AI Predictions** - SAP-RPT-1-OSS powered predictions
5. **🔗 OData** - Connect to SAP systems
6. **🎮 Playground** - Custom model training

## Scaling Configuration

The default configuration:
- **Min Replicas:** 1 (always running)
- **Max Replicas:** 3 (auto-scale under load)
- **CPU:** 2 cores
- **Memory:** 4 GB

To adjust scaling:
```bash
az containerapp update ^
  --name %APP_NAME% ^
  --resource-group %RESOURCE_GROUP% ^
  --min-replicas 1 ^
  --max-replicas 5 ^
  --cpu 4 ^
  --memory 8Gi
```

## Cost Optimization

**Estimated Monthly Cost (East US):**
- Container Apps (1 replica, 2 vCPU, 4GB): ~$60/month
- Container Registry (Basic): ~$5/month
- **Total:** ~$65/month

**To reduce costs:**
1. Set min-replicas to 0 (app sleeps when not in use)
2. Use smaller CPU/memory allocation
3. Delete when not needed:
   ```bash
   az group delete --name %RESOURCE_GROUP% --yes
   ```

## Monitoring & Logs

### View Live Logs
```bash
az containerapp logs show --name %APP_NAME% --resource-group %RESOURCE_GROUP% --follow
```

### View Metrics (Azure Portal)
1. Go to your Container App
2. Click **Monitoring** > **Metrics**
3. View:
   - CPU usage
   - Memory usage
   - Request count
   - Response time

## Updating Your Application

When you make code changes:

```bash
# Build new image
az acr build --registry %ACR_NAME% --image sap-finance-dashboard:latest .

# Update container app
az containerapp update ^
  --name %APP_NAME% ^
  --resource-group %RESOURCE_GROUP% ^
  --image %ACR_NAME%.azurecr.io/sap-finance-dashboard:latest
```

## Troubleshooting

### App not starting?
Check logs:
```bash
az containerapp logs show --name %APP_NAME% --resource-group %RESOURCE_GROUP% --tail 100
```

### Out of memory?
Increase memory allocation:
```bash
az containerapp update --name %APP_NAME% --resource-group %RESOURCE_GROUP% --memory 8Gi
```

### Port issues?
Verify port is set to 7862:
```bash
az containerapp show --name %APP_NAME% --resource-group %RESOURCE_GROUP% --query properties.configuration.ingress
```

### Can't access URL?
Check ingress is enabled:
```bash
az containerapp ingress show --name %APP_NAME% --resource-group %RESOURCE_GROUP%
```

## Security Best Practices

1. **Use Managed Identity** for Azure service connections
2. **Store secrets** in Azure Key Vault
3. **Enable HTTPS only** (default enabled)
4. **Restrict ingress** to specific IPs if needed
5. **Rotate tokens** regularly

## Support

- **Azure Container Apps Docs:** https://learn.microsoft.com/azure/container-apps/
- **SAP-RPT-1-OSS:** https://github.com/SAP-samples/sap-rpt-1-oss
- **Gradio Docs:** https://gradio.app/docs

## Clean Up

To delete all resources and stop charges:
```bash
az group delete --name %RESOURCE_GROUP% --yes --no-wait
```

---

**Ready to deploy?** Run the automated script or follow the manual steps above! 🚀

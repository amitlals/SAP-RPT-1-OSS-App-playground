# Azure Container Apps Deployment Script for SAP Finance Dashboard
# Run this script in PowerShell after logging in to Azure CLI

# ============================================
# CONFIGURATION - Update these values
# ============================================
$RESOURCE_GROUP = "rg-sap-finance-dashboard"
$LOCATION = "eastus"
$ACR_NAME = "acrsapfinance$(Get-Random -Maximum 9999)"  # Must be globally unique
$ENVIRONMENT_NAME = "env-sap-finance"
$APP_NAME = "sap-finance-dashboard"

# Optional: Set your secrets (or configure later in Azure Portal)
$HUGGINGFACE_TOKEN = ""  # Your Hugging Face token
$SAP_USERNAME = ""       # SAP OData username
$SAP_PASSWORD = ""       # SAP OData password

# ============================================
# STEP 1: Login to Azure (if not already logged in)
# ============================================
Write-Host "Step 1: Checking Azure login..." -ForegroundColor Cyan
$account = az account show 2>$null | ConvertFrom-Json
if (-not $account) {
    Write-Host "Please login to Azure..." -ForegroundColor Yellow
    az login
}
Write-Host "Logged in as: $($account.user.name)" -ForegroundColor Green

# ============================================
# STEP 2: Create Resource Group
# ============================================
Write-Host "`nStep 2: Creating Resource Group..." -ForegroundColor Cyan
az group create --name $RESOURCE_GROUP --location $LOCATION
Write-Host "Resource Group '$RESOURCE_GROUP' created in $LOCATION" -ForegroundColor Green

# ============================================
# STEP 3: Create Azure Container Registry
# ============================================
Write-Host "`nStep 3: Creating Azure Container Registry..." -ForegroundColor Cyan
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku Basic --admin-enabled true
Write-Host "Container Registry '$ACR_NAME' created" -ForegroundColor Green

# Get ACR credentials
$ACR_USERNAME = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASSWORD = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv
$ACR_LOGIN_SERVER = az acr show --name $ACR_NAME --query loginServer -o tsv

Write-Host "ACR Login Server: $ACR_LOGIN_SERVER" -ForegroundColor Yellow

# ============================================
# STEP 4: Build and Push Docker Image
# ============================================
Write-Host "`nStep 4: Building and pushing Docker image..." -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow

# Build using ACR Tasks (no local Docker required)
az acr build --registry $ACR_NAME --image "${APP_NAME}:latest" .

Write-Host "Docker image built and pushed to ACR" -ForegroundColor Green

# ============================================
# STEP 5: Create Container Apps Environment
# ============================================
Write-Host "`nStep 5: Creating Container Apps Environment..." -ForegroundColor Cyan
az containerapp env create `
    --name $ENVIRONMENT_NAME `
    --resource-group $RESOURCE_GROUP `
    --location $LOCATION

Write-Host "Container Apps Environment '$ENVIRONMENT_NAME' created" -ForegroundColor Green

# ============================================
# STEP 6: Deploy Container App
# ============================================
Write-Host "`nStep 6: Deploying Container App..." -ForegroundColor Cyan

# Build the secrets parameter if tokens are provided
$secretsParam = ""
$envVarsParam = ""

if ($HUGGINGFACE_TOKEN) {
    $secretsParam = "--secrets huggingface-token=$HUGGINGFACE_TOKEN"
    $envVarsParam = "--env-vars HUGGINGFACE_TOKEN=secretref:huggingface-token"
}

# Create the container app
az containerapp create `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --environment $ENVIRONMENT_NAME `
    --image "${ACR_LOGIN_SERVER}/${APP_NAME}:latest" `
    --registry-server $ACR_LOGIN_SERVER `
    --registry-username $ACR_USERNAME `
    --registry-password $ACR_PASSWORD `
    --target-port 7862 `
    --ingress external `
    --cpu 2 `
    --memory 4Gi `
    --min-replicas 1 `
    --max-replicas 3 `
    --query properties.configuration.ingress.fqdn

Write-Host "`nContainer App deployed!" -ForegroundColor Green

# ============================================
# STEP 7: Get Application URL
# ============================================
Write-Host "`nStep 7: Getting Application URL..." -ForegroundColor Cyan
$APP_URL = az containerapp show `
    --name $APP_NAME `
    --resource-group $RESOURCE_GROUP `
    --query properties.configuration.ingress.fqdn -o tsv

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host "`nYour SAP Finance Dashboard is available at:" -ForegroundColor Cyan
Write-Host "https://$APP_URL" -ForegroundColor Yellow
Write-Host "`n============================================" -ForegroundColor Green

# ============================================
# NEXT STEPS
# ============================================
Write-Host "`nNEXT STEPS:" -ForegroundColor Cyan
Write-Host "1. Configure secrets in Azure Portal:" -ForegroundColor White
Write-Host "   - HUGGINGFACE_TOKEN: Your Hugging Face API token" -ForegroundColor Gray
Write-Host "   - SAP_USERNAME: SAP OData username (optional)" -ForegroundColor Gray
Write-Host "   - SAP_PASSWORD: SAP OData password (optional)" -ForegroundColor Gray
Write-Host "`n2. To update the app after code changes:" -ForegroundColor White
Write-Host "   az acr build --registry $ACR_NAME --image ${APP_NAME}:latest ." -ForegroundColor Gray
Write-Host "   az containerapp update --name $APP_NAME --resource-group $RESOURCE_GROUP --image ${ACR_LOGIN_SERVER}/${APP_NAME}:latest" -ForegroundColor Gray
Write-Host "`n3. To view logs:" -ForegroundColor White
Write-Host "   az containerapp logs show --name $APP_NAME --resource-group $RESOURCE_GROUP --follow" -ForegroundColor Gray
Write-Host "`n4. To delete all resources:" -ForegroundColor White
Write-Host "   az group delete --name $RESOURCE_GROUP --yes --no-wait" -ForegroundColor Gray

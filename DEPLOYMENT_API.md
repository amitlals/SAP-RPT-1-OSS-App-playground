# 🏢 SAP RPT-1-OSS API Deployment Guide

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Azure Container App                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SAP RPT-1-OSS API                        ││
│  │                   (FastAPI + TabPFN)                        ││
│  │                                                              ││
│  │  Endpoints:                                                  ││
│  │  • POST /predict/sales-status                                ││
│  │  • POST /predict/revenue                                     ││
│  │  • POST /predict/profitability                               ││
│  │  • POST /predict/classification                              ││
│  │  • POST /predict/regression                                  ││
│  │  • GET  /health                                              ││
│  │  • GET  /docs (Swagger UI)                                   ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                           │
│  • Sales Order Status Prediction                                 │
│  • Revenue Forecasting                                           │
│  • Profitability Analysis                                        │
│  • Custom Predictions                                            │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (Local)

### 1. Install Dependencies
```bash
pip install fastapi uvicorn tabpfn-client pandas numpy scikit-learn streamlit requests
```

### 2. Start API Server
```bash
python sap_rpt1_api.py
```
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

### 3. Start Frontend (Optional)
```bash
streamlit run sap_rpt1_frontend.py
```
- Frontend: http://localhost:8501

---

## Deploy to Azure Container Apps

### Step 1: Create Dockerfile for API
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sap_rpt1_api.py .

EXPOSE 8000

CMD ["uvicorn", "sap_rpt1_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Create requirements.txt
```
fastapi>=0.100.0
uvicorn>=0.23.0
tabpfn-client>=0.1.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pydantic>=2.0.0
```

### Step 3: Build and Push Image
```bash
# Login to Azure
az login

# Create Container Registry
az acr create --resource-group sap-rpt1-secure-rg --name saprpt1acr --sku Basic

# Login to ACR
az acr login --name saprpt1acr

# Build and push
az acr build --registry saprpt1acr --image sap-rpt1-api:v1 .
```

### Step 4: Deploy Container App
```bash
# Create Container Apps Environment
az containerapp env create \
  --name sap-rpt1-env \
  --resource-group sap-rpt1-secure-rg \
  --location eastus2

# Deploy the API
az containerapp create \
  --name sap-rpt1-api \
  --resource-group sap-rpt1-secure-rg \
  --environment sap-rpt1-env \
  --image saprpt1acr.azurecr.io/sap-rpt1-api:v1 \
  --target-port 8000 \
  --ingress external \
  --registry-server saprpt1acr.azurecr.io \
  --min-replicas 1 \
  --max-replicas 3
```

### Step 5: Get API URL
```bash
az containerapp show \
  --name sap-rpt1-api \
  --resource-group sap-rpt1-secure-rg \
  --query "properties.configuration.ingress.fqdn" -o tsv
```

---

## API Usage Examples

### Health Check
```bash
curl https://sap-rpt1-api.<region>.azurecontainerapps.io/health
```

### Predict Sales Order Status
```bash
curl -X POST https://sap-rpt1-api.<region>.azurecontainerapps.io/predict/sales-status \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [{
      "region": "North",
      "product_category": "Electronics",
      "quantity": 50,
      "unit_price": 299.99,
      "discount_pct": 10,
      "days_to_deliver": 7,
      "customer_rating": 4.5,
      "previous_orders": 12
    }]
  }'
```

### Predict Revenue
```bash
curl -X POST https://sap-rpt1-api.<region>.azurecontainerapps.io/predict/revenue \
  -H "Content-Type: application/json" \
  -d '{
    "orders": [{
      "region": "West",
      "product_category": "Industrial",
      "quantity": 100,
      "unit_price": 450.00,
      "discount_pct": 15,
      "customer_rating": 4.0,
      "previous_orders": 25
    }]
  }'
```

### Predict Profitability
```bash
curl -X POST https://sap-rpt1-api.<region>.azurecontainerapps.io/predict/profitability \
  -H "Content-Type: application/json" \
  -d '{
    "periods": [{
      "revenue": 850000,
      "cogs": 380000,
      "operating_expenses": 280000,
      "depreciation": 35000,
      "interest_expense": 18000,
      "tax_rate": 0.25
    }]
  }'
```

---

## Integration with SAP

### OData Connector Example
```python
import requests
from sap_odata_connector import SAPODataClient

# Connect to SAP S/4HANA
sap_client = SAPODataClient(
    base_url="https://sap-server.com/sap/opu/odata/sap/",
    username="SAP_USER",
    password="SAP_PASS"
)

# Fetch sales orders
orders = sap_client.get("API_SALES_ORDER_SRV/A_SalesOrder")

# Predict status for each order
for order in orders:
    response = requests.post(
        "https://sap-rpt1-api.azurecontainerapps.io/predict/sales-status",
        json={"orders": [format_order(order)]}
    )
    prediction = response.json()["predictions"][0]
    print(f"Order {order['SalesOrder']}: {prediction}")
```

---

## Files

| File | Description |
|------|-------------|
| `sap_rpt1_api.py` | FastAPI REST API server |
| `sap_rpt1_frontend.py` | Streamlit web frontend |
| `Dockerfile` | Container image definition |
| `requirements.txt` | Python dependencies |

---

## Next Steps

1. ✅ Run locally and test all endpoints
2. ⬜ Deploy API to Azure Container Apps
3. ⬜ Configure HuggingFace token in Container App secrets
4. ⬜ Connect to real SAP OData endpoints
5. ⬜ Add authentication (Azure AD / API keys)
6. ⬜ Set up monitoring with Application Insights

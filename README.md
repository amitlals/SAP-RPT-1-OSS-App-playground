# SAP RPT-1-OSS | AI-Powered Tabular ML

**Enterprise-grade machine learning predictions for SAP data using In-Context Learning**

🔗 **Live Demo**: [https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io/](https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io/)

---

## 🎯 Overview

This application provides AI-powered predictions for SAP enterprise data using the [TabPFN](https://www.tabpfn.com/) model - a foundation model for tabular data that uses **In-Context Learning** (no traditional training required).

### Key Capabilities

| Use Case | Description |
|----------|-------------|
| **Sales Order Status** | Predict delivery status (On-Time, Delayed, Cancelled) |
| **Revenue Forecasting** | Forecast order amounts and financial metrics |
| **Profitability Analysis** | Classify accounts as Profitable or Loss-making |
| **Custom Predictions** | Generic classification/regression on any tabular data |

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
│                              │   (PriorLabs)       │        │
│                              └─────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/amitlals/SAP-RPT-1-OSS-App-playground.git
cd SAP-RPT-1-OSS-App-playground

# Install dependencies
pip install -r requirements.txt

# Set TabPFN token
python -c "from tabpfn_client import set_access_token; set_access_token('YOUR_TOKEN')"

# Start API server
uvicorn sap_rpt1_api:app --host 0.0.0.0 --port 8000

# Start Streamlit (new terminal)
streamlit run sap_rpt1_frontend.py --server.port 8501
```

### Get TabPFN Token

1. Go to [tabpfn.com](https://www.tabpfn.com/)
2. Sign up for free account
3. Get your API token

---

## 📁 Project Structure

```
SAP-RPT-1-OSS-App/
├── sap_rpt1_api.py        # FastAPI REST API
├── sap_rpt1_frontend.py   # Streamlit web UI
├── Dockerfile.azure       # Container image for Azure
├── supervisord.conf       # Process manager config
├── start.sh               # Container startup script
├── requirements.txt       # Python dependencies
├── data/                  # Sample SAP datasets
├── models/                # Model utilities
└── utils/                 # Helper functions
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/predict/sales-status` | Predict order delivery status |
| POST | `/predict/revenue` | Forecast order amounts |
| POST | `/predict/profitability` | Predict financial profitability |
| POST | `/predict/classification` | Generic classification |
| POST | `/predict/regression` | Generic regression |

---

## ☁️ Azure Deployment

Deployed on **Azure Container Apps** with:

| Component | Value |
|-----------|-------|
| **Resource Group** | sap-rpt1-secure-rg |
| **Container Registry** | saprpt1acr.azurecr.io |
| **Container App** | sap-rpt1-oss-app |
| **Region** | East US 2 |
| **Scaling** | 1-3 replicas |

See [DEPLOYMENT_API.md](DEPLOYMENT_API.md) for full deployment instructions.

---

## 👤 Author

**Amit Lal**  
🔗 [aka.ms/amitlal](https://aka.ms/amitlal)

---

## 📄 License

Apache 2.0

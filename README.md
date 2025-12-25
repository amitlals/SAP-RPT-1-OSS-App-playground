# 🚀 SAP RPT-1-OSS | AI-Powered Enterprise ML Suite

**Three production-ready applications demonstrating SAP-RPT-1-OSS's In-Context Learning for SAP enterprise data**

[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Spaces-yellow)](https://huggingface.co/amitgpt)
[![Azure](https://img.shields.io/badge/☁️-Azure%20Container%20Apps-blue)](https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📦 Applications

| # | App | Description | Live Demo |
|---|-----|-------------|-----------|
| 1 | **Finance Dashboard** | SAP financial statement analysis & profitability prediction | [🤗 HuggingFace](https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS) |
| 2 | **Forecast Showdown** | RPT-1 vs LLM comparison on numeric forecasting | [🤗 HuggingFace](https://huggingface.co/spaces/amitgpt/sap-rpt1-forecast-showdown)|
| 3 | **Predictive Integrity** | Proactive SAP operations risk prediction (Jobs, Transports, Interfaces) | [🤗 HuggingFace](https://huggingface.co/spaces/amitgpt/sap-predictive-integrity-using-RPT-1) |
| 4 | **SAP Local RPT-1 Workspace** | SAP-RPT-1-OSS on Microsoft Foundry Hosted |[☁️ Azure](https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io/) |

---
<img width="1124" height="629" alt="image" src="https://github.com/user-attachments/assets/fe9bf40e-100f-4b11-95e1-b79255e47f68" /> <br>
<img width="1100" height="583" alt="image" src="https://github.com/user-attachments/assets/b2a8e0f5-ee1b-45e0-bf1a-64ce9d57e8a2" />


## 🏗️ Repository Structure

```
SAP-RPT-1-OSS-App/
│
├── 📁 apps/                          # Application packages
│   ├── 01-finance-dashboard/         # App 1: Financial Analysis
│   │   ├── app.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── README.md
│   │
│   ├── 02-forecast-showdown/         # App 2: RPT-1 vs LLM
│   │   ├── app.py
│   │   ├── api.py
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── README.md
│   │
│   └── 03-predictive-integrity/      # App 3: Operations Risk
│       ├── app.py
│       ├── utils/
│       ├── requirements.txt
│       ├── Dockerfile
│       └── README.md
│
├── 📁 deploy/                        # Deployment configurations
│   ├── azure/                        # Azure Container Apps
│   │   ├── Dockerfile
│   │   ├── deploy-azure.ps1
│   │   ├── supervisord.conf
│   │   └── README.md
│   │
│   └── huggingface/                  # HuggingFace Spaces
│       └── README.md
│
├── 📁 shared/                        # Shared utilities
│   ├── data/                         # Sample datasets
│   ├── models/                       # Model utilities
│   └── utils/                        # Common helpers
│
├── 📁 docs/                          # Documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   └── DEPLOYMENT_GUIDE.md
│
└── README.md                         # This file
```

---

## 🎯 Technology Stack

| Component | Technology |
|-----------|------------|
| **ML Model** | [RPT-1](https://github.com/SAP-samples/sap-rpt-1-oss) - Foundation Model for Tabular Data |
| **Frontend** | Streamlit |
| **Backend** | FastAPI |
| **Cloud** | Azure Container Apps, HuggingFace Spaces |
| **Container** | Docker |

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
python --version

# Get TabPFN token from https://tabpfn.com
export TABPFN_ACCESS_TOKEN="your_token_here"
```

### Run Locally

```bash
# Clone the repo
git clone https://github.com/amitlals/SAP-RPT-1-OSS-App-playground.git
cd SAP-RPT-1-OSS-App-playground

# Install dependencies
pip install -r requirements.txt

# Run any app
streamlit run apps/01-finance-dashboard/app.py
streamlit run apps/02-forecast-showdown/app.py
streamlit run apps/03-predictive-integrity/app.py
```

---

## 🤗 Deploy to HuggingFace

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** as SDK
3. Copy the contents of the desired app folder
4. Add secrets:
   - `TABPFN_ACCESS_TOKEN` - From [tabpfn.com](https://tabpfn.com)
   - `SAP_RPT1_TOKEN` (optional) - For SAP-RPT-1 Closed API

See [deploy/huggingface/README.md](deploy/huggingface/README.md) for details.

---

## ☁️ Deploy to Azure

```powershell
# From repository root
cd deploy/azure
./deploy-azure.ps1
```

See [deploy/azure/README.md](deploy/azure/README.md) for details.

---

## 📊 Sample Data

All apps use synthetic SAP-like datasets:

| Dataset | Description | SAP Tables Referenced |
|---------|-------------|----------------------|
| `synthetic_financial_statements.csv` | Balance sheet & income data | BSEG, BKPF, T001 |
| `synthetic_gl_accounts.csv` | General ledger accounts | SKA1, SKAT |
| `synthetic_sales_orders.csv` | Sales order history | VBAK, VBAP, LIKP |

---

## 🏷️ SAP Table References

### App 3: Predictive Integrity

| Scenario | SAP Tables | Key Fields |
|----------|------------|------------|
| **Job Failure** | TBTCO, TBTCP | JOBNAME, STATUS, SDLUNAME |
| **Transport Failure** | E070, E071 | TRKORR, TRSTATUS, AS4USER |
| **Interface Failure** | EDIDC, EDIDS | DOCNUM, STATUS, MESTYP |

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [PriorLabs](https://priorlabs.ai/) - TabPFN creators
- [SAP](https://www.sap.com/) - Enterprise software inspiration
- [Streamlit](https://streamlit.io/) - UI framework

---

**⚠️ Disclaimer**: This project uses synthetic data for demonstration purposes. SAP® is a registered trademark of SAP SE. This project is not affiliated with or endorsed by SAP SE.

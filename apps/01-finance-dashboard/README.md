# 📊 SAP Finance Dashboard

**AI-Powered Financial Statement Analysis using SAP-RPT-1-OSS (TabPFN)**

🔗 **Live Demo**: [https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS](https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS)

---

## 🎯 Overview

Interactive dashboard for analyzing SAP financial statements using In-Context Learning. Provides profitability predictions and revenue forecasting without traditional ML training.

### Features

| Feature | Description |
|---------|-------------|
| **Profitability Prediction** | Classify accounts as Profitable/Loss-making |
| **Revenue Forecasting** | Predict order amounts and financial metrics |
| **Interactive Charts** | Real-time visualization of predictions |
| **Data Export** | Download predictions as CSV |

---

## 🚀 Quick Start

### Local Development

```bash
cd apps/01-finance-dashboard

# Install dependencies
pip install -r requirements.txt

# Set TabPFN token
export TABPFN_ACCESS_TOKEN="your_token_here"

# Run the app
streamlit run app.py
```

### HuggingFace Deployment

1. Create a new Space on HuggingFace
2. Copy contents of this folder
3. Add `TABPFN_ACCESS_TOKEN` secret
4. Space will auto-build and deploy

---

## 📁 Files

```
01-finance-dashboard/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

---

## 🔧 Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `TABPFN_ACCESS_TOKEN` | TabPFN API token from [tabpfn.com](https://tabpfn.com) | Yes |

---

## 📊 Sample Data

Uses synthetic SAP financial data including:
- `synthetic_financial_statements.csv` - Balance sheet & income data
- `synthetic_gl_accounts.csv` - General ledger accounts
- `synthetic_sales_orders.csv` - Sales order history

---

## 🏷️ SAP Table References

| Synthetic Column | SAP Table | SAP Field |
|-----------------|-----------|-----------|
| COMPANY_CODE | T001 | BUKRS |
| FISCAL_YEAR | T009 | GJAHR |
| GL_ACCOUNT | SKA1 | SAKNR |
| PROFIT_CENTER | CEPC | PRCTR |

---

*Powered by [SAP-RPT-1-OSS](https://sap.com/rpt-1) - Foundation Model for Tabular Data*

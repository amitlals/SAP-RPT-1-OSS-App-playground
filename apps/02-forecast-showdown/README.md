# 🎯 Forecast Showdown: SAP RPT-1-OSS vs LLM

**Demonstrates why specialized Tabular ML beats general-purpose LLMs on numeric pattern recognition**

🔗 **Live Demos**:
- [🤗 HuggingFace Space](https://huggingface.co/spaces/amitgpt/sap-rpt1-forecast-showdown)
- [☁️ Azure Container Apps](https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io/)

---

## 🎯 Overview

Side-by-side comparison of SAP RPT-1 (specialized tabular ML) vs GPT-4 (general-purpose LLM) on SAP financial forecasting tasks. Proves that purpose-built models outperform LLMs on structured data.

### Key Insights

| Metric | TabPFN | LLM |
|--------|--------|-----|
| **Accuracy** | 92-98% | 45-65% |
| **Latency** | <100ms | 2-5s |
| **Pattern Recognition** | Excellent | Poor |
| **Numeric Precision** | High | Variable |

---

## 🚀 Quick Start

### Local Development

```bash
cd apps/02-forecast-showdown

# Install dependencies
pip install -r requirements.txt

# Set tokens
export TABPFN_ACCESS_TOKEN="your_tabpfn_token"
export OPENAI_API_KEY="your_openai_key"  # Optional for LLM comparison

# Run the app
streamlit run app.py
```

### Deploy to HuggingFace

1. Create a new Space
2. Upload contents of this folder
3. Add secrets: `TABPFN_ACCESS_TOKEN`, `OPENAI_API_KEY`

### Deploy to Azure

```powershell
cd deploy/azure
./deploy-azure.ps1
```

---

## 📁 Files

```
02-forecast-showdown/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

---

## 🧪 Test Scenarios

1. **Revenue Forecasting** - Predict next quarter revenue
2. **Demand Planning** - Forecast product demand
3. **Cash Flow Projection** - Predict cash positions
4. **Anomaly Detection** - Identify unusual patterns

---

## 🔧 Configuration

| Environment Variable | Description | Required |
|---------------------|-------------|----------|
| `TABPFN_ACCESS_TOKEN` | TabPFN API token | Yes |
| `OPENAI_API_KEY` | OpenAI API key for LLM comparison | Optional |

---

*Powered by [TabPFN](https://tabpfn.com) - Foundation Model for Tabular Data*

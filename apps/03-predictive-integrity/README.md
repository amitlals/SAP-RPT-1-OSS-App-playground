# 🛡️ SAP Predictive Integrity

**Proactive SAP Operations Risk Prediction using SAP-RPT-1**

🔗 **Live Demo**: [https://huggingface.co/spaces/amitgpt/sap-predictive-integrity-using-RPT-1](https://huggingface.co/spaces/amitgpt/sap-predictive-integrity-using-RPT-1)

---

## 🎯 Overview

Predict operational failures in SAP systems before they occur. Uses synthetic data modeled after real SAP tables to demonstrate proactive risk detection.

### Failure Scenarios

| Scenario | SAP Tables | Description |
|----------|------------|-------------|
| **Job Failure** | TBTCO, TBTCP | Background job failures based on scheduling patterns |
| **Transport Failure** | E070, E071 | Change transport issues based on object complexity |
| **Interface Failure** | EDIDC, EDIDS | IDoc/EDI communication failures |

---

## 🚀 Quick Start

### Local Development

```bash
cd apps/03-predictive-integrity

# Install dependencies
pip install -r requirements.txt

# Set token (choose one)
export TABPFN_ACCESS_TOKEN="your_token"  # For SAP-RPT-1-OSS (public)
# OR
export SAP_RPT1_TOKEN="your_token"  # For SAP-RPT-1 (closed API)

# Run the app
streamlit run app.py
```

### HuggingFace Deployment

1. Create a new Space on HuggingFace
2. Upload contents of this folder
3. Add secrets:
   - `TABPFN_ACCESS_TOKEN` for public model
   - `SAP_RPT1_TOKEN` for closed API (optional)

---

## 📁 Files

```
03-predictive-integrity/
├── app.py                        # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── failure_data_generator.py # Synthetic SAP data generation
│   └── sap_rpt1_client.py        # API clients (OSS + Closed)
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
└── README.md                     # This file
```

---

## 🏷️ SAP Table References

### Job Failure (TBTCO/TBTCP)

| Synthetic Column | SAP Table | SAP Field | Description |
|-----------------|-----------|-----------|-------------|
| JOBNAME | TBTCO | JOBNAME | Background job name |
| STATUS | TBTCO | STATUS | Job status (A/S/R/F) |
| STRTDATE | TBTCO | STRTDATE | Scheduled start date |
| SDLUNAME | TBTCO | SDLUNAME | User who scheduled job |
| JOBCLASS | TBTCO | JOBCLASS | Job priority class |

### Transport Failure (E070/E071)

| Synthetic Column | SAP Table | SAP Field | Description |
|-----------------|-----------|-----------|-------------|
| TRKORR | E070 | TRKORR | Transport request number |
| TRSTATUS | E070 | TRSTATUS | Transport status |
| AS4USER | E070 | AS4USER | Owner of transport |
| TRFUNCTION | E070 | TRFUNCTION | Transport type |

### Interface Failure (EDIDC/EDIDS)

| Synthetic Column | SAP Table | SAP Field | Description |
|-----------------|-----------|-----------|-------------|
| DOCNUM | EDIDC | DOCNUM | IDoc document number |
| STATUS | EDIDC | STATUS | IDoc status |
| MESTYP | EDIDC | MESTYP | Message type |
| SNDPRT | EDIDC | SNDPRT | Sender partner type |

---

## 🔧 Model Options

| Model | Type | Token Source |
|-------|------|--------------|
| **SAP-RPT-1-OSS** | Public (TabPFN) | [tabpfn.com](https://tabpfn.com) |
| **SAP-RPT-1** | Closed API | [rpt.cloud.sap](https://rpt.cloud.sap/docs) |

---

## ⚠️ Security Notes

- Treat API tokens like passwords
- Never commit tokens to source control
- Use environment variables or secrets management
- Rotate tokens periodically

---

*Powered by [TabPFN](https://tabpfn.com) - Foundation Model for Tabular Data*

**Disclaimer**: SAP® is a registered trademark of SAP SE. This project uses synthetic data and is not affiliated with SAP SE.

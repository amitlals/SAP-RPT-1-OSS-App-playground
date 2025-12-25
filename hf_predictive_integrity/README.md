---
title: SAP Predictive Integrity
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Proactive SAP Operational Risk Prediction with SAP-RPT-1
---

# 🛡️ SAP Predictive Integrity

**Proactive Operational Risk Prediction for SAP Systems using SAP-RPT-1 Tabular ML**

This interactive demo predicts operational failures in SAP environments using synthetic datasets that mirror real SAP table structures.

## 🎯 Prediction Scenarios

| Scenario | SAP Tables Referenced | Risk Factors |
|----------|----------------------|--------------|
| 🔮 **Job Failure** | TBTCO, TBTCP, TBTCS | Concurrency, Memory, Delay, Job Class |
| 📦 **Transport Risk** | E070, E071, TPLOG | Object Count, Author Success Rate, System Load |
| 🔗 **Interface Health** | EDIDC, EDIDS, ARFCSSTATE | Queue Depth, Partner Reliability, Payload Size |

## 🤖 Models Supported

- **SAP-RPT-1-OSS (Public)**: Open-source tabular ML via TabPFN on HuggingFace
- **SAP-RPT-1 (Closed API)**: Enterprise API with Bearer token authentication
- **Offline Mode**: Mock predictions for demo purposes

## ✨ Features

- **1,000 Row Analysis**: Score 1,000 synthetic SAP records per scenario
- **Seed Rotation**: Regenerate datasets with different random seeds
- **Drift Detection**: Alerts when data distribution shifts significantly
- **Confidence Scoring**: Each prediction includes probability confidence
- **Remediation Playbooks**: Actionable guidance for HIGH risk entities
- **Export**: Download scored CSV and audit JSON

## 📊 Dataset Schema

Each scenario generates synthetic data mimicking real SAP table structures:

### Job Failure (TBTCO/TBTCP)
| Column | Source | Description |
|--------|--------|-------------|
| JOBNAME | TBTCO | Background job name |
| JOBCLASS | TBTCO | Priority (A/B/C) |
| DURATION_SEC | Derived | Job execution time |
| CONCURRENT_JOBS | Synthetic | Jobs running simultaneously |
| MEM_USAGE_PCT | Synthetic | Memory consumption |
| RISK_SCORE | Computed | Weighted risk metric |
| RISK_LABEL | Computed | HIGH/MEDIUM/LOW classification |

### Transport Failure (E070/E071)
| Column | Source | Description |
|--------|--------|-------------|
| TRKORR | E070 | Transport request number |
| OBJ_COUNT | E071 | Number of objects |
| AUTHOR_SUCCESS_RATE | Synthetic | Historical author success |
| TARGET_SYS_LOAD | Synthetic | Target system load |

### Interface Failure (EDIDC/EDIDS)
| Column | Source | Description |
|--------|--------|-------------|
| MESTYP | EDIDC | IDoc message type |
| QUEUE_DEPTH | Synthetic | Queue backlog |
| PARTNER_RELIABILITY | Synthetic | Partner success rate |

## 🚀 How to Use

1. **Select Model Type**: Choose SAP-RPT-1-OSS (public) or SAP-RPT-1 (closed API)
2. **Connect**: Validate your connection or use offline mode
3. **Generate Data**: Select scenario and generate 1,000 synthetic rows
4. **Score**: Run predictions with batch processing
5. **Analyze**: Review top 100 high-risk entities with remediation guidance
6. **Export**: Download results as CSV or JSON audit log

## 💡 Key Insight

> **RISK_SCORE and RISK_LABEL are synthetic labels computed for demonstration purposes.** In production, replace these with actual historical outcomes from your SAP system.

---

**Developed by [Amit Lal](https://aka.ms/amitlal)**

⚖️ **Disclaimer:** SAP, SAP RPT, SAP-RPT-1, and all SAP logos and product names are trademarks or registered trademarks of SAP SE in Germany and other countries. This is an independent demonstration project for educational purposes only and is not affiliated with, endorsed by, or sponsored by SAP SE or any enterprise. The synthetic datasets used in this application are for demonstration purposes only and do not represent real SAP system data. All other trademarks are the property of their respective owners.

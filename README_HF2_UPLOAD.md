# 🚀 Hugging Face Upload Instructions - SAP Predictive Integrity

This document provides step-by-step instructions for deploying the **SAP Predictive Integrity** app to Hugging Face Spaces.

---

## 📁 Files to Upload

Upload the entire `hf_predictive_integrity/` folder to your HuggingFace Space:

```
hf_predictive_integrity/
├── README.md           # HF Space metadata & description
├── Dockerfile          # Docker build configuration
├── requirements.txt    # Python dependencies
├── app.py              # Main Streamlit application
└── utils/
    ├── __init__.py
    ├── failure_data_generator.py
    └── sap_rpt1_client.py
```

---

## 🛠️ Step-by-Step Deployment

### Option 1: Using HuggingFace Web UI

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure:
   - **Space name**: `sap-predictive-integrity`
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU Basic (free tier works)
4. Click **"Create Space"**
5. Upload all files from `hf_predictive_integrity/` folder
6. Wait for build to complete (~2-3 minutes)

### Option 2: Using Git

```bash
# Clone your new space
git clone https://huggingface.co/spaces/YOUR_USERNAME/sap-predictive-integrity
cd sap-predictive-integrity

# Copy files from hf_predictive_integrity folder
cp -r /path/to/hf_predictive_integrity/* .

# Push to HuggingFace
git add .
git commit -m "Initial deployment"
git push
```

### Option 3: Using HuggingFace CLI

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create and upload
huggingface-cli repo create sap-predictive-integrity --type space --space_sdk docker
huggingface-cli upload YOUR_USERNAME/sap-predictive-integrity ./hf_predictive_integrity --repo-type space
```

---

## 🔧 Configuration Notes

### Docker SDK
The app uses Docker SDK because it requires `tabpfn-client` which has specific dependencies. The Dockerfile:
- Uses Python 3.11 slim image
- Installs all requirements
- Exposes port 7860 (HF standard)
- Runs Streamlit in headless mode

### Environment Variables (Optional)
If you want to pre-configure a HuggingFace token for TabPFN:
1. Go to your Space settings
2. Add secret: `TABPFN_ACCESS_TOKEN` = your HF token

### Hardware Requirements
- **CPU Basic** (free): Works fine for demo purposes
- **CPU Upgrade**: Recommended for faster TabPFN inference

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] App loads without errors
- [ ] Header displays with gradient styling
- [ ] 3 tabs are visible (Setup & Data, Prediction, Insights & Export)
- [ ] "Run in Offline Mode" button works
- [ ] Data generation produces 1,000 rows
- [ ] Mock predictions run successfully
- [ ] Export buttons download CSV/JSON files
- [ ] Footer and disclaimer display correctly

---

## 🔗 Live Demo URL

After deployment, your app will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/sap-predictive-integrity
```

---

## 📊 App Features

| Feature | Description |
|---------|-------------|
| 🔮 Job Failure | Predict SAP background job failures (TBTCO/TBTCP) |
| 📦 Transport Risk | Predict transport import failures (E070/E071) |
| 🔗 Interface Health | Predict IDoc/RFC failures (EDIDC/EDIDS) |
| 🤖 SAP-RPT-1-OSS | HuggingFace TabPFN integration |
| 📈 1,000 Row Analysis | Score 1,000 synthetic records per run |
| 📋 Remediation Playbooks | Actionable guidance for high-risk entities |

---

## ⚖️ Disclaimer

SAP, SAP RPT, SAP-RPT-1, and all SAP logos and product names are trademarks or registered trademarks of SAP SE in Germany and other countries. This is an independent demonstration project for educational purposes only and is not affiliated with, endorsed by, or sponsored by SAP SE or any enterprise.

---

**Developed by [Amit Lal](https://aka.ms/amitlal)** | December 2025

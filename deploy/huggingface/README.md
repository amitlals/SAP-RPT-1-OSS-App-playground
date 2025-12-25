# 🤗 HuggingFace Spaces Deployment

**Deploy SAP RPT-1-OSS applications to HuggingFace Spaces**

---

## 🎯 Overview

Each app can be deployed as a standalone HuggingFace Space using Docker SDK.

---

## 🚀 Quick Deploy

### Step 1: Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name your space (e.g., `sap-finance-dashboard`)
3. Select **Docker** as the SDK
4. Choose **Blank** template
5. Set visibility (Public/Private)

### Step 2: Upload App Files

Upload the contents of the desired app folder:

| App | Source Folder |
|-----|---------------|
| Finance Dashboard | `apps/01-finance-dashboard/` |
| Forecast Showdown | `apps/02-forecast-showdown/` |
| Predictive Integrity | `apps/03-predictive-integrity/` |

Required files:
- `app.py`
- `requirements.txt`
- `Dockerfile`
- `README.md`
- `utils/` (if present)

### Step 3: Configure Secrets

Go to **Settings → Repository secrets** and add:

| Secret | Description | Required |
|--------|-------------|----------|
| `TABPFN_ACCESS_TOKEN` | Token from [tabpfn.com](https://tabpfn.com) | Yes |
| `SAP_RPT1_TOKEN` | SAP-RPT-1 closed API token | Optional |
| `OPENAI_API_KEY` | OpenAI key for LLM comparison | Optional |

### Step 4: Wait for Build

The Space will automatically:
1. Clone your files
2. Build the Docker image
3. Start the container
4. Expose on port 7860

---

## 📦 Dockerfile Template

All apps use this base Dockerfile:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

---

## 🔧 Space Configuration

Create a `README.md` with YAML frontmatter:

```yaml
---
title: SAP Finance Dashboard
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
```

---

## 🌐 Live Spaces

| App | HuggingFace URL |
|-----|-----------------|
| Finance Dashboard | [huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS](https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS) |
| Forecast Showdown | [huggingface.co/spaces/amitgpt/sap-rpt1-forecast-showdown](https://huggingface.co/spaces/amitgpt/sap-rpt1-forecast-showdown) |
| Predictive Integrity | [huggingface.co/spaces/amitgpt/sap-predictive-integrity-using-RPT-1](https://huggingface.co/spaces/amitgpt/sap-predictive-integrity-using-RPT-1) |

---

## 🐛 Troubleshooting

### Build Fails

Check the build logs in the **Logs** tab. Common issues:
- Missing dependencies in `requirements.txt`
- Incorrect Python version
- Port mismatch (must use 7860)

### App Crashes

1. Check runtime logs
2. Ensure all secrets are configured
3. Verify file paths are relative

### Slow Startup

First startup takes 2-5 minutes for:
- Docker image build
- Dependency installation
- Model initialization

---

## 💡 Tips

1. **Use `.gitignore`** - Exclude `__pycache__`, `.env`, `venv/`
2. **Pin versions** - Specify exact versions in `requirements.txt`
3. **Optimize image** - Use `--no-cache-dir` for pip
4. **Test locally** - Run `docker build . && docker run -p 7860:7860`

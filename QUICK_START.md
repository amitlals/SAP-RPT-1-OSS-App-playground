# 🚀 SAP Finance Dashboard - Quick Setup Guide

## ✅ Status: Live & Ready!

Your SAP Finance Dashboard is **fully deployed** on Hugging Face Spaces at:
### 🔗 https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS

---

## 🔐 One Final Step: Enable Model Features (5 minutes)

The dashboard is **fully functional** but needs your HF authentication to use AI prediction features.

### Quick Setup (3 clicks):

#### Step 1️⃣: Accept Model Access
- Go to: https://huggingface.co/SAP/sap-rpt-1-oss
- Click the blue **"Agree"** button
- ✓ Done

#### Step 2️⃣: Create Access Token
- Visit: https://huggingface.co/settings/tokens
- Click **"New token"**
- Name: `sap-rpt-oss-access`
- Type: **"Read"**
- Click **"Create token"**
- 📋 Copy the token (long string starting with `hf_`)

#### Step 3️⃣: Add to Your Space
- Go to your Space: https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS
- Click **⚙ Settings** (top right)
- Find **"Repository secrets"**
- Click **"Add secret"**
  - Name: `HF_TOKEN`
  - Value: Paste your token
- Click **"Add secret"**
- Wait 1-2 minutes for rebuild

#### ✨ Done!
When the rebuild finishes, refresh the Space and all features work!

---

## 📚 What Each Tab Does

| Tab | Function | Status |
|-----|----------|--------|
| **Dashboard** | Financial metrics & charts | ✅ Works |
| **Data Explorer** | Browse datasets | ✅ Works |
| **Upload** | Upload custom data | ✅ Works |
| **OData** | Connect to SAP systems | ✅ Works |
| **Predictions** | AI-powered forecasts | ⏳ Needs HF token |
| **Playground** | Train & test models | ⏳ Needs HF token |

---

## 🛠️ How the Auth Works

1. You set `HF_TOKEN` in HF Spaces secrets
2. Space rebuilds with token available
3. Your app automatically logs in to Hugging Face
4. Model downloads from gated repository
5. Model cached for fast access

**Your token is ONLY used to download the model. No data is uploaded.**

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| Still see 401 error | Wait 2+ min for rebuild, refresh browser |
| "HF_TOKEN not found" | Check secret name is exactly `HF_TOKEN` (case-sensitive) |
| Model still won't load | Verify you clicked "Agree" on model page |
| Old app still showing | Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac) |

---

## 📖 Full Documentation

For detailed setup, troubleshooting, and security info:
- See: `HF_AUTHENTICATION_SETUP.md` in the repo

---

## 🎉 That's It!

Your SAP Finance Dashboard is ready. With one authentication token, you'll unlock full AI functionality.

**Questions?** Check the troubleshooting section or read `HF_AUTHENTICATION_SETUP.md` for detailed guidance.

---

*Powered by Gradio • HuggingFace Spaces • SAP-RPT-1-OSS*

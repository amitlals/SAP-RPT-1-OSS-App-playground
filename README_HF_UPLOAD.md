---
title: SAP RPT-1-OSS Forecast Integrity Showdown
emoji: 🏆
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
short_description: SAP RPT-1-OSS vs LLMs on Tabular Data
---

# 🏆 SAP RPT-1-OSS Forecast Integrity Showdown

**Can ChatGPT beat specialized ML on spreadsheet data? Let's find out!**

This interactive demo compares **SAP RPT-1-OSS** (a specialized tabular ML model) against popular LLMs on complex numeric pattern recognition tasks.

## 🎯 What This Demo Shows

| Tab | Task | Features | Classes |
|-----|------|----------|---------|
| 📊 Forecast Integrity | Financial risk classification | 13 | 3 (HIGH/MEDIUM/LOW) |
| 🏦 Credit Risk | Credit rating prediction | 19 | 6 (AAA to B) |
| 👤 Customer Churn | Churn prediction | 20 | 4 (CHURNED to STABLE) |

## 🤖 Models Compared

- **SAP RPT-1-OSS**: Specialized tabular ML via TabPFN
- **Groq Llama 3.3 8B**: Fast open-source LLM
- **Groq Llama 3.3 70B**: Larger reasoning model
- **Google Gemini 2.0 Flash**: Google's fast multimodal
- **Mistral Small**: Efficient European LLM
- **OpenRouter Claude**: Anthropic's Claude via OpenRouter

## 💡 Key Insight

> **Specialized ML models like SAP RPT-1-OSS consistently outperform general-purpose LLMs on tabular/numeric data because they understand numbers as mathematical values, not text tokens.**

## 🚀 Try It

1. Select an LLM challenger
2. Adjust sample size and seed
3. Click "Run Showdown"
4. Watch SAP RPT-1-OSS dominate!

---

**Developed by [Amit Lal](https://aka.ms/amitlal)**

⚖️ *Disclaimer: SAP, SAP RPT, and all SAP logos and product names are trademarks or registered trademarks of SAP SE in Germany and other countries. This is an independent demonstration project for educational purposes only and is not affiliated with, endorsed by, or sponsored by SAP SE or any enterprise. All other trademarks are the property of their respective owners.*

# 📡 API Reference

**SAP RPT-1-OSS REST API Documentation**

---

## 🌐 Base URLs

| Environment | URL |
|-------------|-----|
| Local | `http://localhost:8000` |
| Azure | `https://sap-rpt1-oss-app.wonderfulground-a83887c1.eastus2.azurecontainerapps.io` |

---

## 🔐 Authentication

All prediction endpoints require a valid TabPFN token.

```bash
# Set token via environment variable
export TABPFN_ACCESS_TOKEN="your_token_here"
```

---

## 📋 Endpoints

### Health Check

```http
GET /health
```

**Response**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "tabpfn_connected": true
}
```

---

### Predict Sales Status

Predict delivery status for sales orders.

```http
POST /predict/sales-status
Content-Type: application/json
```

**Request Body**

```json
{
  "data": [
    {
      "ORDER_ID": "SO-2024-001",
      "CUSTOMER_ID": "CUST-100",
      "ORDER_VALUE": 15000.00,
      "PRODUCT_CATEGORY": "Electronics",
      "REGION": "North America",
      "LEAD_TIME_DAYS": 7
    }
  ]
}
```

**Response**

```json
{
  "predictions": [
    {
      "ORDER_ID": "SO-2024-001",
      "predicted_status": "On-Time",
      "confidence": 0.92,
      "risk_factors": ["none"]
    }
  ],
  "model": "TabPFN",
  "latency_ms": 145
}
```

---

### Predict Revenue

Forecast revenue for financial planning.

```http
POST /predict/revenue
Content-Type: application/json
```

**Request Body**

```json
{
  "data": [
    {
      "PERIOD": "2024-Q1",
      "PRODUCT_LINE": "Software",
      "HISTORICAL_REVENUE": [100000, 120000, 115000],
      "MARKET_GROWTH": 0.05,
      "SEASONALITY_INDEX": 1.2
    }
  ]
}
```

**Response**

```json
{
  "predictions": [
    {
      "PERIOD": "2024-Q1",
      "predicted_revenue": 138000.00,
      "confidence_interval": {
        "lower": 125000.00,
        "upper": 151000.00
      }
    }
  ],
  "model": "TabPFN",
  "latency_ms": 98
}
```

---

### Predict Profitability

Classify accounts as profitable or loss-making.

```http
POST /predict/profitability
Content-Type: application/json
```

**Request Body**

```json
{
  "data": [
    {
      "ACCOUNT_ID": "ACC-001",
      "COMPANY_CODE": "1000",
      "REVENUE": 500000,
      "COSTS": 420000,
      "OVERHEAD": 50000,
      "MARKET_SEGMENT": "Enterprise"
    }
  ]
}
```

**Response**

```json
{
  "predictions": [
    {
      "ACCOUNT_ID": "ACC-001",
      "classification": "Profitable",
      "probability": 0.87,
      "margin_estimate": 0.06
    }
  ],
  "model": "TabPFN",
  "latency_ms": 112
}
```

---

### Generic Classification

Custom classification on any tabular data.

```http
POST /predict/classify
Content-Type: application/json
```

**Request Body**

```json
{
  "train_data": [
    {"feature1": 1.0, "feature2": "A", "label": "positive"},
    {"feature1": 2.0, "feature2": "B", "label": "negative"}
  ],
  "test_data": [
    {"feature1": 1.5, "feature2": "A"}
  ],
  "target_column": "label"
}
```

**Response**

```json
{
  "predictions": [
    {
      "predicted_label": "positive",
      "probabilities": {
        "positive": 0.78,
        "negative": 0.22
      }
    }
  ],
  "model": "TabPFN",
  "latency_ms": 156
}
```

---

## 🚨 Error Responses

### 400 Bad Request

```json
{
  "error": "Invalid request payload",
  "details": "Missing required field: ORDER_ID"
}
```

### 401 Unauthorized

```json
{
  "error": "Invalid or missing API token",
  "details": "Set TABPFN_ACCESS_TOKEN environment variable"
}
```

### 429 Rate Limited

```json
{
  "error": "Rate limit exceeded",
  "retryAfter": 30
}
```

### 500 Internal Server Error

```json
{
  "error": "Prediction failed",
  "details": "TabPFN service unavailable"
}
```

---

## 📊 Rate Limits

| Tier | Requests/Min | Rows/Request |
|------|--------------|--------------|
| Free | 60 | 1,000 |
| Pro | 300 | 10,000 |
| Enterprise | Unlimited | 100,000 |

---

## 💡 Best Practices

1. **Batch requests** - Send multiple rows in one request
2. **Handle 429s** - Implement exponential backoff
3. **Validate locally** - Check data before sending
4. **Use async** - For high-throughput applications

---

## 🔧 SDK Usage

### Python

```python
from tabpfn_client import TabPFNClassifier

# Initialize
clf = TabPFNClassifier()

# Train and predict
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
probabilities = clf.predict_proba(X_test)
```

### cURL

```bash
curl -X POST https://api.tabpfn.com/predict \
  -H "Authorization: Bearer $TABPFN_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"data": [{"col1": 1, "col2": 2}]}'
```

---

## 📚 Related Documentation

- [TabPFN Documentation](https://docs.tabpfn.com)
- [Architecture Overview](ARCHITECTURE.md)
- [Deployment Guide](DEPLOYMENT_GUIDE.md)

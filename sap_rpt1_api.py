"""
🏢 SAP RPT-1-OSS API Server
============================
FastAPI-based REST API for SAP Tabular ML Predictions

Endpoints:
- POST /predict/classification - Classify SAP data
- POST /predict/regression - Predict numeric values
- POST /predict/sales-status - Predict sales order status
- POST /predict/profitability - Predict financial profitability
- GET /health - Health check
- GET /docs - Swagger UI

Run: uvicorn sap_rpt1_api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import TabPFN
try:
    from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="SAP RPT-1-OSS API",
    description="""
    ## AI-Powered Tabular ML for SAP Enterprise Data
    
    This API provides machine learning predictions for SAP enterprise data using 
    **In-Context Learning** - no traditional model training required.
    
    ### Capabilities:
    - **Classification**: Predict categories (order status, account types, etc.)
    - **Regression**: Predict numeric values (revenue, costs, etc.)
    - **SAP Use Cases**: Sales orders, GL accounts, financial statements
    
    ### Technology:
    - Model: SAP RPT-1-OSS (TabPFN-based)
    - Platform: Azure ML Secure Workspace
    - Framework: FastAPI
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Generic prediction request"""
    train_data: List[Dict[str, Any]] = Field(..., description="Training data as list of dictionaries")
    train_labels: List[Any] = Field(..., description="Training labels")
    predict_data: List[Dict[str, Any]] = Field(..., description="Data to predict on")
    
    class Config:
        json_schema_extra = {
            "example": {
                "train_data": [
                    {"feature1": 100, "feature2": 50},
                    {"feature1": 200, "feature2": 75}
                ],
                "train_labels": [0, 1],
                "predict_data": [
                    {"feature1": 150, "feature2": 60}
                ]
            }
        }

class SalesOrderRequest(BaseModel):
    """Sales order prediction request"""
    orders: List[Dict[str, Any]] = Field(..., description="Sales orders to predict status for")
    
    class Config:
        json_schema_extra = {
            "example": {
                "orders": [
                    {
                        "region": "North",
                        "product_category": "Electronics",
                        "quantity": 50,
                        "unit_price": 299.99,
                        "discount_pct": 10,
                        "days_to_deliver": 7,
                        "customer_rating": 4.5,
                        "previous_orders": 12
                    }
                ]
            }
        }

class FinancialRequest(BaseModel):
    """Financial profitability prediction request"""
    periods: List[Dict[str, Any]] = Field(..., description="Financial periods to predict")
    
    class Config:
        json_schema_extra = {
            "example": {
                "periods": [
                    {
                        "revenue": 850000,
                        "cogs": 380000,
                        "operating_expenses": 280000,
                        "depreciation": 35000,
                        "interest_expense": 18000,
                        "tax_rate": 0.25
                    }
                ]
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    predictions: List[Any]
    model: str = "SAP-RPT-1-OSS"
    task_type: str
    confidence: Optional[List[float]] = None
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_available: bool
    version: str
    endpoints: List[str]

# ============================================================================
# Helper Functions
# ============================================================================

def generate_training_data():
    """Generate synthetic SAP training data"""
    np.random.seed(42)
    n = 80  # Training samples
    
    # Sales order training data
    sales_train = pd.DataFrame({
        'region': np.random.choice([0, 1, 2, 3, 4], n),  # Encoded regions
        'product_category': np.random.choice([0, 1, 2, 3], n),
        'quantity': np.random.randint(1, 100, n),
        'unit_price': np.random.uniform(50, 500, n),
        'discount_pct': np.random.uniform(0, 25, n),
        'days_to_deliver': np.random.randint(3, 45, n),
        'customer_rating': np.random.uniform(1, 5, n),
        'previous_orders': np.random.randint(0, 50, n)
    })
    sales_labels = np.random.choice([0, 1, 2, 3], n, p=[0.5, 0.25, 0.15, 0.10])  # Delivered, In Process, Cancelled, Open
    
    # Financial training data
    fin_train = pd.DataFrame({
        'revenue': np.random.uniform(500000, 1000000, n),
        'cogs': np.random.uniform(250000, 450000, n),
        'operating_expenses': np.random.uniform(150000, 350000, n),
        'depreciation': np.random.uniform(20000, 50000, n),
        'interest_expense': np.random.uniform(10000, 30000, n),
        'tax_rate': np.random.uniform(0.20, 0.30, n)
    })
    # Calculate profitability
    gross_profit = fin_train['revenue'] - fin_train['cogs']
    ebitda = gross_profit - fin_train['operating_expenses']
    net_income = (ebitda - fin_train['depreciation'] - fin_train['interest_expense']) * (1 - fin_train['tax_rate'])
    fin_labels = (net_income > 0).astype(int)
    
    return {
        'sales': (sales_train, sales_labels),
        'financial': (fin_train, fin_labels)
    }

# Pre-generate training data
TRAINING_DATA = generate_training_data()

# Status mapping
STATUS_MAP = {0: 'Delivered', 1: 'In Process', 2: 'Cancelled', 3: 'Open'}
REGION_MAP = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Central': 4}
PRODUCT_MAP = {'Electronics': 0, 'Industrial': 1, 'Consumer': 2, 'Services': 3}

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """API root - redirect to docs"""
    return {
        "message": "SAP RPT-1-OSS API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model availability"""
    return HealthResponse(
        status="healthy",
        model_available=TABPFN_AVAILABLE,
        version="1.0.0",
        endpoints=[
            "/predict/classification",
            "/predict/regression", 
            "/predict/sales-status",
            "/predict/profitability"
        ]
    )

@app.post("/predict/classification", response_model=PredictionResponse, tags=["Predictions"])
async def predict_classification(request: PredictionRequest):
    """
    Generic classification prediction.
    
    Provide training data with labels and data to predict.
    Uses In-Context Learning - no pre-training required.
    """
    if not TABPFN_AVAILABLE:
        raise HTTPException(status_code=503, detail="TabPFN model not available")
    
    try:
        # Convert to DataFrames
        X_train = pd.DataFrame(request.train_data)
        y_train = np.array(request.train_labels)
        X_predict = pd.DataFrame(request.predict_data)
        
        # Train and predict
        classifier = TabPFNClassifier()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_predict)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            task_type="classification",
            metadata={
                "train_samples": len(X_train),
                "predict_samples": len(X_predict),
                "features": list(X_train.columns)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/regression", response_model=PredictionResponse, tags=["Predictions"])
async def predict_regression(request: PredictionRequest):
    """
    Generic regression prediction.
    
    Provide training data with numeric labels and data to predict.
    """
    if not TABPFN_AVAILABLE:
        raise HTTPException(status_code=503, detail="TabPFN model not available")
    
    try:
        X_train = pd.DataFrame(request.train_data)
        y_train = np.array(request.train_labels)
        X_predict = pd.DataFrame(request.predict_data)
        
        regressor = TabPFNRegressor()
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_predict)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            task_type="regression",
            metadata={
                "train_samples": len(X_train),
                "predict_samples": len(X_predict)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/sales-status", response_model=PredictionResponse, tags=["SAP Use Cases"])
async def predict_sales_status(request: SalesOrderRequest):
    """
    Predict SAP Sales Order Status.
    
    Predicts: Delivered, In Process, Cancelled, or Open
    
    Uses pre-trained context from synthetic SAP data.
    """
    if not TABPFN_AVAILABLE:
        raise HTTPException(status_code=503, detail="TabPFN model not available")
    
    try:
        # Get training data
        X_train, y_train = TRAINING_DATA['sales']
        
        # Prepare prediction data
        predict_data = []
        for order in request.orders:
            predict_data.append({
                'region': REGION_MAP.get(order.get('region', 'North'), 0),
                'product_category': PRODUCT_MAP.get(order.get('product_category', 'Electronics'), 0),
                'quantity': order.get('quantity', 1),
                'unit_price': order.get('unit_price', 100),
                'discount_pct': order.get('discount_pct', 0),
                'days_to_deliver': order.get('days_to_deliver', 7),
                'customer_rating': order.get('customer_rating', 3),
                'previous_orders': order.get('previous_orders', 0)
            })
        
        X_predict = pd.DataFrame(predict_data)
        
        # Predict
        classifier = TabPFNClassifier()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_predict)
        
        # Map to status names
        status_predictions = [STATUS_MAP[p] for p in predictions]
        
        return PredictionResponse(
            predictions=status_predictions,
            task_type="classification",
            metadata={
                "use_case": "SAP Sales Order Status",
                "possible_values": list(STATUS_MAP.values()),
                "orders_processed": len(request.orders)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/profitability", response_model=PredictionResponse, tags=["SAP Use Cases"])
async def predict_profitability(request: FinancialRequest):
    """
    Predict Financial Period Profitability.
    
    Predicts: Profitable or Loss based on financial metrics.
    
    Uses pre-trained context from synthetic SAP financial data.
    """
    if not TABPFN_AVAILABLE:
        raise HTTPException(status_code=503, detail="TabPFN model not available")
    
    try:
        # Get training data
        X_train, y_train = TRAINING_DATA['financial']
        
        # Prepare prediction data
        predict_data = []
        for period in request.periods:
            predict_data.append({
                'revenue': period.get('revenue', 500000),
                'cogs': period.get('cogs', 300000),
                'operating_expenses': period.get('operating_expenses', 200000),
                'depreciation': period.get('depreciation', 30000),
                'interest_expense': period.get('interest_expense', 15000),
                'tax_rate': period.get('tax_rate', 0.25)
            })
        
        X_predict = pd.DataFrame(predict_data)
        
        # Predict
        classifier = TabPFNClassifier()
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_predict)
        
        # Map to labels
        profit_predictions = ['Profitable' if p == 1 else 'Loss' for p in predictions]
        
        return PredictionResponse(
            predictions=profit_predictions,
            task_type="classification",
            metadata={
                "use_case": "SAP Financial Profitability",
                "possible_values": ["Profitable", "Loss"],
                "periods_processed": len(request.periods)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/revenue", response_model=PredictionResponse, tags=["SAP Use Cases"])
async def predict_revenue(request: SalesOrderRequest):
    """
    Predict Sales Order Revenue.
    
    Predicts total order amount based on order characteristics.
    """
    if not TABPFN_AVAILABLE:
        raise HTTPException(status_code=503, detail="TabPFN model not available")
    
    try:
        # Generate training data for regression
        np.random.seed(42)
        n = 80
        X_train = pd.DataFrame({
            'region': np.random.choice([0, 1, 2, 3, 4], n),
            'product_category': np.random.choice([0, 1, 2, 3], n),
            'quantity': np.random.randint(1, 100, n),
            'unit_price': np.random.uniform(50, 500, n),
            'discount_pct': np.random.uniform(0, 25, n),
            'customer_rating': np.random.uniform(1, 5, n),
            'previous_orders': np.random.randint(0, 50, n)
        })
        y_train = (X_train['quantity'] * X_train['unit_price'] * (1 - X_train['discount_pct']/100)).values
        
        # Prepare prediction data
        predict_data = []
        for order in request.orders:
            predict_data.append({
                'region': REGION_MAP.get(order.get('region', 'North'), 0),
                'product_category': PRODUCT_MAP.get(order.get('product_category', 'Electronics'), 0),
                'quantity': order.get('quantity', 1),
                'unit_price': order.get('unit_price', 100),
                'discount_pct': order.get('discount_pct', 0),
                'customer_rating': order.get('customer_rating', 3),
                'previous_orders': order.get('previous_orders', 0)
            })
        
        X_predict = pd.DataFrame(predict_data)
        
        # Predict
        regressor = TabPFNRegressor()
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_predict)
        
        return PredictionResponse(
            predictions=[round(p, 2) for p in predictions],
            task_type="regression",
            metadata={
                "use_case": "SAP Revenue Prediction",
                "currency": "USD",
                "orders_processed": len(request.orders)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🏢 SAP RPT-1-OSS API Server")
    print("="*60)
    print(f"\n📍 API Docs: http://localhost:8000/docs")
    print(f"📍 Health:   http://localhost:8000/health")
    print(f"\n🔧 Model Available: {TABPFN_AVAILABLE}")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

"""
🏢 SAP RPT-1-OSS Demo Frontend
===============================
Modern Streamlit UI for SAP AI Predictions

Run: streamlit run sap_rpt1_frontend.py
"""

import streamlit as st
import requests
import pandas as pd
import json
import os
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SAP RPT-1-OSS | AI Predictions",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #0066cc, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #666;
        font-size: 1.2rem;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .profitable {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .loss {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/SAP_2011_logo.svg", width=120)
    st.markdown("---")
    
    st.markdown("### 🎯 Use Cases")
    use_case = st.radio(
        "Select Demo",
        ["🏠 Overview", "📦 Sales Order Status", "💰 Revenue Prediction", "📈 Profitability Analysis", "🔧 Custom Prediction"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ API Status")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"✅ API Online")
        st.info(f"Model: {'Ready' if health['model_available'] else 'Not Available'}")
    except:
        st.error("❌ API Offline")
        st.warning("Start the API server first")
    
    st.markdown("---")
    st.markdown("### 📍 Deployment")
    st.code("""
Platform: Azure ML
Workspace: sap-rpt1-secure-ws
Model: sap-rpt-1-oss
    """)

# ============================================================================
# Main Content
# ============================================================================

# Header
st.markdown('<p class="main-header">SAP RPT-1-OSS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Tabular ML for Enterprise Data</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# Overview Page
# ============================================================================

if use_case == "🏠 Overview":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "In-Context Learning", "No Training Required")
    with col2:
        st.metric("Data Size", "10-1000 rows", "Small Data Ready")
    with col3:
        st.metric("Inference", "< 2 sec", "Real-time")
    with col4:
        st.metric("Tasks", "2 Types", "Classification & Regression")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Available Use Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 📦 Sales Order Status Prediction
        Predict the delivery status of SAP sales orders:
        - **Delivered** - Order completed
        - **In Process** - Order being fulfilled
        - **Cancelled** - Order cancelled
        - **Open** - Order pending
        
        **Input**: Order details (region, product, quantity, price, etc.)
        """)
        
        st.markdown("""
        #### 💰 Revenue Prediction
        Forecast order revenue based on:
        - Product category
        - Quantity and pricing
        - Customer history
        - Discount rates
        """)
    
    with col2:
        st.markdown("""
        #### 📈 Profitability Analysis
        Predict if a financial period will be profitable:
        - Revenue vs COGS
        - Operating expenses
        - Tax implications
        - Net income forecast
        
        **Output**: Profitable / Loss prediction
        """)
        
        st.markdown("""
        #### 🔧 Custom Predictions
        Build your own prediction:
        - Upload training data
        - Define labels
        - Get predictions instantly
        - No model training needed
        """)
    
    st.markdown("---")
    st.markdown("### 🔗 API Endpoints")
    
    endpoints = pd.DataFrame({
        "Endpoint": ["/predict/sales-status", "/predict/revenue", "/predict/profitability", "/predict/classification", "/predict/regression"],
        "Method": ["POST", "POST", "POST", "POST", "POST"],
        "Description": ["Predict order delivery status", "Forecast order revenue", "Predict period profitability", "Generic classification", "Generic regression"]
    })
    st.dataframe(endpoints, use_container_width=True, hide_index=True)

# ============================================================================
# Sales Order Status
# ============================================================================

elif use_case == "📦 Sales Order Status":
    st.markdown("### 📦 Sales Order Status Prediction")
    st.markdown("Predict whether an order will be **Delivered**, **In Process**, **Cancelled**, or **Open**")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Order Details")
        
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
        product = st.selectbox("Product Category", ["Electronics", "Industrial", "Consumer", "Services"])
        quantity = st.slider("Quantity", 1, 100, 25)
        unit_price = st.number_input("Unit Price ($)", 50.0, 500.0, 199.99)
        discount = st.slider("Discount (%)", 0, 25, 10)
        days = st.slider("Days to Deliver", 3, 45, 14)
        rating = st.slider("Customer Rating", 1.0, 5.0, 4.0, 0.5)
        prev_orders = st.number_input("Previous Orders", 0, 100, 5)
        
        predict_btn = st.button("🔮 Predict Status", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### Prediction Result")
        
        if predict_btn:
            with st.spinner("Analyzing order..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/sales-status",
                        json={
                            "orders": [{
                                "region": region,
                                "product_category": product,
                                "quantity": quantity,
                                "unit_price": unit_price,
                                "discount_pct": discount,
                                "days_to_deliver": days,
                                "customer_rating": rating,
                                "previous_orders": prev_orders
                            }]
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["predictions"][0]
                        
                        # Color based on status
                        colors = {
                            "Delivered": "#28a745",
                            "In Process": "#ffc107", 
                            "Cancelled": "#dc3545",
                            "Open": "#17a2b8"
                        }
                        
                        st.markdown(f"""
                        <div style="background: {colors.get(prediction, '#6c757d')}; 
                                    color: white; 
                                    padding: 30px; 
                                    border-radius: 10px; 
                                    text-align: center;
                                    font-size: 2rem;
                                    font-weight: bold;">
                            {prediction}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("#### 📊 Order Summary")
                        st.json({
                            "order_value": f"${quantity * unit_price * (1 - discount/100):,.2f}",
                            "region": region,
                            "product": product,
                            "predicted_status": prediction
                        })
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the server is running.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# Revenue Prediction
# ============================================================================

elif use_case == "💰 Revenue Prediction":
    st.markdown("### 💰 Revenue Prediction")
    st.markdown("Forecast total order revenue based on order characteristics")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Order Parameters")
        
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"], key="rev_region")
        product = st.selectbox("Product Category", ["Electronics", "Industrial", "Consumer", "Services"], key="rev_product")
        quantity = st.slider("Quantity", 1, 100, 30, key="rev_qty")
        unit_price = st.number_input("Unit Price ($)", 50.0, 500.0, 249.99, key="rev_price")
        discount = st.slider("Discount (%)", 0, 25, 5, key="rev_disc")
        rating = st.slider("Customer Rating", 1.0, 5.0, 4.5, 0.5, key="rev_rating")
        prev_orders = st.number_input("Previous Orders", 0, 100, 10, key="rev_prev")
        
        predict_btn = st.button("💵 Predict Revenue", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### Revenue Forecast")
        
        if predict_btn:
            with st.spinner("Calculating revenue..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/revenue",
                        json={
                            "orders": [{
                                "region": region,
                                "product_category": product,
                                "quantity": quantity,
                                "unit_price": unit_price,
                                "discount_pct": discount,
                                "customer_rating": rating,
                                "previous_orders": prev_orders
                            }]
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["predictions"][0]
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; 
                                    padding: 30px; 
                                    border-radius: 10px; 
                                    text-align: center;">
                            <div style="font-size: 1rem; opacity: 0.8;">Predicted Revenue</div>
                            <div style="font-size: 2.5rem; font-weight: bold;">${prediction:,.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Comparison with simple calculation
                        simple_calc = quantity * unit_price * (1 - discount/100)
                        
                        st.markdown("---")
                        st.markdown("#### 📊 Comparison")
                        
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.metric("AI Prediction", f"${prediction:,.2f}")
                        with comp_col2:
                            st.metric("Simple Calculation", f"${simple_calc:,.2f}", 
                                     f"{((prediction - simple_calc) / simple_calc * 100):+.1f}%")
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API")

# ============================================================================
# Profitability Analysis
# ============================================================================

elif use_case == "📈 Profitability Analysis":
    st.markdown("### 📈 Profitability Analysis")
    st.markdown("Predict if a financial period will be profitable or result in a loss")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Financial Metrics")
        
        revenue = st.number_input("Revenue ($)", 100000, 2000000, 850000, step=10000)
        cogs = st.number_input("Cost of Goods Sold ($)", 50000, 1000000, 380000, step=10000)
        opex = st.number_input("Operating Expenses ($)", 50000, 500000, 280000, step=10000)
        depreciation = st.number_input("Depreciation ($)", 5000, 100000, 35000, step=1000)
        interest = st.number_input("Interest Expense ($)", 0, 50000, 18000, step=1000)
        tax_rate = st.slider("Tax Rate (%)", 15, 35, 25) / 100
        
        predict_btn = st.button("📊 Analyze Profitability", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### Profitability Prediction")
        
        if predict_btn:
            with st.spinner("Analyzing financials..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/profitability",
                        json={
                            "periods": [{
                                "revenue": revenue,
                                "cogs": cogs,
                                "operating_expenses": opex,
                                "depreciation": depreciation,
                                "interest_expense": interest,
                                "tax_rate": tax_rate
                            }]
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result["predictions"][0]
                        
                        is_profitable = prediction == "Profitable"
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {'#11998e, #38ef7d' if is_profitable else '#eb3349, #f45c43'}); 
                                    color: white; 
                                    padding: 30px; 
                                    border-radius: 10px; 
                                    text-align: center;">
                            <div style="font-size: 3rem;">{'✅' if is_profitable else '❌'}</div>
                            <div style="font-size: 2rem; font-weight: bold;">{prediction}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show calculations
                        st.markdown("---")
                        st.markdown("#### 📊 Financial Breakdown")
                        
                        gross_profit = revenue - cogs
                        ebitda = gross_profit - opex
                        ebit = ebitda - depreciation
                        ebt = ebit - interest
                        net_income = ebt * (1 - tax_rate)
                        
                        fin_df = pd.DataFrame({
                            "Metric": ["Gross Profit", "EBITDA", "EBIT", "EBT", "Net Income"],
                            "Value": [f"${gross_profit:,.0f}", f"${ebitda:,.0f}", f"${ebit:,.0f}", f"${ebt:,.0f}", f"${net_income:,.0f}"]
                        })
                        st.dataframe(fin_df, use_container_width=True, hide_index=True)
                    else:
                        st.error(f"API Error: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API")

# ============================================================================
# Custom Prediction
# ============================================================================

elif use_case == "🔧 Custom Prediction":
    st.markdown("### 🔧 Custom Prediction")
    st.markdown("Build your own prediction with custom training data")
    
    tab1, tab2 = st.tabs(["Classification", "Regression"])
    
    with tab1:
        st.markdown("#### Upload Training Data")
        
        # Example data
        example_train = pd.DataFrame({
            "feature1": [100, 200, 150, 300, 250],
            "feature2": [50, 60, 55, 70, 65],
            "feature3": [1.5, 2.0, 1.8, 2.5, 2.2]
        })
        example_labels = [0, 1, 0, 1, 1]
        
        st.markdown("**Training Data (CSV format)**")
        train_data = st.text_area("Enter training data as CSV", 
                                  example_train.to_csv(index=False), 
                                  height=150)
        
        labels = st.text_input("Labels (comma-separated)", "0, 1, 0, 1, 1")
        
        st.markdown("**Prediction Data**")
        pred_data = st.text_area("Data to predict on (CSV)", 
                                 "feature1,feature2,feature3\n175,58,1.9",
                                 height=80)
        
        if st.button("🔮 Classify", type="primary"):
            try:
                import io
                train_df = pd.read_csv(io.StringIO(train_data))
                pred_df = pd.read_csv(io.StringIO(pred_data))
                label_list = [int(x.strip()) for x in labels.split(",")]
                
                response = requests.post(
                    f"{API_URL}/predict/classification",
                    json={
                        "train_data": train_df.to_dict('records'),
                        "train_labels": label_list,
                        "predict_data": pred_df.to_dict('records')
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🎯 Predictions: {result['predictions']}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>SAP RPT-1-OSS | Azure ML Secure Workspace | Model: sap-rpt-1-oss</p>
    <p style="font-size: 0.9rem;">Developed by <a href="https://aka.ms/amitlal" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: bold;">Amit Lal</a></p>
    <p style="font-size: 0.8rem;">Powered by In-Context Learning | © 2024</p>
</div>
""", unsafe_allow_html=True)

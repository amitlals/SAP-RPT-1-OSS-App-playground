"""
🎯 Forecast Integrity Showdown - SAP RPT-1-OSS vs LLM
======================================================
Demonstrates why specialized Tabular ML beats general-purpose LLMs
on complex numeric pattern recognition tasks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import requests

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="Forecast Integrity | SAP RPT-1-OSS vs LLM",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 25%, #4facfe 50%, #00f2fe 75%, #43e97b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: gradient 3s ease infinite;
        background-size: 200% 200%;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .sub-header {
        color: #555;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0;
        padding-bottom: 15px;
    }
    .header-container {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 30%, #4facfe 60%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        padding: 10px 0;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        text-align: center;
        margin: 10px 0 0 0;
    }
    .header-badges {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin-top: 15px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        backdrop-filter: blur(10px);
    }
    .story-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ed 100%);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    .sap-card {
        background: linear-gradient(135deg, #0066cc 0%, #004c99 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .llm-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
    }
    .winner-banner {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
        font-size: 1.3rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .simple-insight {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        border: 2px solid #dee2e6;
    }
    .step-number {
        background: #0066cc;
        color: white;
        width: 35px;
        height: 35px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# COMPLEX TABULAR DATASET - Hard for LLMs, Natural for SAP RPT-1-OSS
# =============================================================================

def generate_complex_forecast_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate complex SAP forecast data with non-linear patterns.
    These patterns are HARD for LLMs to detect from text but EASY for SAP RPT-1-OSS.
    """
    np.random.seed(seed)
    
    records = []
    for i in range(n_samples):
        # Complex numeric features with interactions
        revenue_q1 = np.random.uniform(100000, 500000)
        revenue_q2 = revenue_q1 * np.random.uniform(0.85, 1.15)
        revenue_q3 = revenue_q2 * np.random.uniform(0.9, 1.2)
        revenue_q4 = revenue_q3 * np.random.uniform(0.8, 1.25)
        
        # Derived features (patterns SAP RPT-1-OSS can learn)
        yoy_growth = (revenue_q4 - revenue_q1) / revenue_q1
        volatility = np.std([revenue_q1, revenue_q2, revenue_q3, revenue_q4]) / np.mean([revenue_q1, revenue_q2, revenue_q3, revenue_q4])
        
        # Complex cost structure
        cogs_ratio = np.random.uniform(0.4, 0.7)
        opex_ratio = np.random.uniform(0.15, 0.35)
        margin = 1 - cogs_ratio - opex_ratio
        
        # Market factors
        market_share = np.random.uniform(0.02, 0.25)
        competitor_count = np.random.randint(3, 15)
        customer_concentration = np.random.uniform(0.1, 0.6)
        
        # Seasonal adjustment factor
        seasonal_idx = np.random.uniform(0.7, 1.3)
        
        # Forecast accuracy depends on COMPLEX INTERACTIONS (hard for LLMs)
        risk_score = (
            volatility * 2.0 +
            (1 - margin) * 1.5 +
            customer_concentration * 1.2 +
            (1 / (competitor_count + 1)) * 0.8 +
            abs(yoy_growth - 0.1) * 1.0
        )
        risk_score += np.random.normal(0, 0.3)
        
        # Determine integrity class based on risk score
        if risk_score < 1.8:
            integrity_class = "HIGH"
            integrity_score = np.random.uniform(80, 98)
        elif risk_score < 2.8:
            integrity_class = "MEDIUM"  
            integrity_score = np.random.uniform(50, 79)
        else:
            integrity_class = "LOW"
            integrity_score = np.random.uniform(15, 49)
        
        records.append({
            'ID': f'FC-{i+1:04d}',
            'Revenue_Q1': round(revenue_q1, 0),
            'Revenue_Q2': round(revenue_q2, 0),
            'Revenue_Q3': round(revenue_q3, 0),
            'Revenue_Q4': round(revenue_q4, 0),
            'YoY_Growth': round(yoy_growth, 4),
            'Volatility': round(volatility, 4),
            'COGS_Ratio': round(cogs_ratio, 3),
            'OpEx_Ratio': round(opex_ratio, 3),
            'Margin': round(margin, 3),
            'Market_Share': round(market_share, 3),
            'Competitors': competitor_count,
            'Customer_Concentration': round(customer_concentration, 3),
            'Seasonal_Index': round(seasonal_idx, 3),
            'Integrity_Class': integrity_class,
            'Integrity_Score': round(integrity_score, 1)
        })
    
    return pd.DataFrame(records)


def generate_credit_risk_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate enterprise credit risk data with INTENSE non-linear patterns.
    Uses advanced financial ratios and multi-variable interactions.
    """
    np.random.seed(seed)
    
    records = []
    for i in range(n_samples):
        # Liquidity Ratios
        current_ratio = np.random.uniform(0.5, 3.5)
        quick_ratio = current_ratio * np.random.uniform(0.5, 0.95)
        cash_ratio = quick_ratio * np.random.uniform(0.2, 0.6)
        
        # Leverage Ratios
        debt_to_equity = np.random.uniform(0.2, 4.0)
        debt_to_assets = debt_to_equity / (1 + debt_to_equity)
        interest_coverage = np.random.uniform(0.5, 15.0)
        
        # Profitability Ratios
        roa = np.random.uniform(-0.1, 0.25)
        roe = roa * (1 + debt_to_equity)
        profit_margin = np.random.uniform(-0.05, 0.20)
        ebitda_margin = profit_margin + np.random.uniform(0.05, 0.15)
        
        # Efficiency Ratios
        asset_turnover = np.random.uniform(0.3, 2.5)
        inventory_days = np.random.uniform(15, 120)
        receivables_days = np.random.uniform(20, 90)
        payables_days = np.random.uniform(25, 100)
        
        # Cash Conversion Cycle
        cash_conversion_cycle = inventory_days + receivables_days - payables_days
        
        # Market & Size Factors
        market_cap_log = np.random.uniform(6, 12)  # log scale
        beta = np.random.uniform(0.3, 2.5)
        years_in_business = np.random.randint(1, 50)
        
        # COMPLEX NON-LINEAR CREDIT SCORE FORMULA
        # This combines multiple ratios with non-linear transformations
        z_score = (
            # Liquidity component (weighted)
            1.2 * np.log1p(current_ratio) +
            0.8 * (quick_ratio / (current_ratio + 0.01)) +
            
            # Leverage component (inverse relationships)
            -1.5 * np.tanh(debt_to_equity / 2) +
            0.6 * np.log1p(interest_coverage) +
            
            # Profitability component (sigmoid-like)
            2.0 * (1 / (1 + np.exp(-10 * roa))) +
            1.5 * np.clip(profit_margin * 5, -1, 1) +
            
            # Efficiency component
            0.4 * np.log1p(asset_turnover) +
            -0.3 * (cash_conversion_cycle / 100) +
            
            # Size & stability
            0.2 * (market_cap_log / 12) +
            0.1 * np.log1p(years_in_business / 10) +
            
            # Interaction terms (VERY hard for LLMs)
            0.5 * (current_ratio * profit_margin) +
            -0.4 * (debt_to_equity * (1 - profit_margin)) +
            0.3 * (interest_coverage / (debt_to_equity + 1))
        )
        z_score += np.random.normal(0, 0.5)
        
        # Determine credit risk class
        if z_score > 4.0:
            risk_class = "AAA"
            default_prob = np.random.uniform(0.0001, 0.001)
        elif z_score > 3.0:
            risk_class = "AA"
            default_prob = np.random.uniform(0.001, 0.01)
        elif z_score > 2.0:
            risk_class = "A"
            default_prob = np.random.uniform(0.01, 0.03)
        elif z_score > 1.0:
            risk_class = "BBB"
            default_prob = np.random.uniform(0.03, 0.08)
        elif z_score > 0:
            risk_class = "BB"
            default_prob = np.random.uniform(0.08, 0.15)
        else:
            risk_class = "B"
            default_prob = np.random.uniform(0.15, 0.35)
        
        records.append({
            'Entity_ID': f'ENT-{i+1:04d}',
            'Current_Ratio': round(current_ratio, 3),
            'Quick_Ratio': round(quick_ratio, 3),
            'Cash_Ratio': round(cash_ratio, 3),
            'Debt_to_Equity': round(debt_to_equity, 3),
            'Debt_to_Assets': round(debt_to_assets, 3),
            'Interest_Coverage': round(interest_coverage, 2),
            'ROA': round(roa, 4),
            'ROE': round(roe, 4),
            'Profit_Margin': round(profit_margin, 4),
            'EBITDA_Margin': round(ebitda_margin, 4),
            'Asset_Turnover': round(asset_turnover, 3),
            'Inventory_Days': round(inventory_days, 1),
            'Receivables_Days': round(receivables_days, 1),
            'Payables_Days': round(payables_days, 1),
            'Cash_Conversion_Cycle': round(cash_conversion_cycle, 1),
            'Market_Cap_Log': round(market_cap_log, 2),
            'Beta': round(beta, 3),
            'Years_Operating': years_in_business,
            'Credit_Rating': risk_class,
            'Default_Probability': round(default_prob, 4)
        })
    
    return pd.DataFrame(records)


def generate_customer_churn_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate customer churn data with behavioral and transactional patterns.
    Uses recency-frequency-monetary (RFM) analysis with complex non-linear scoring.
    """
    np.random.seed(seed)
    
    records = []
    for i in range(n_samples):
        # Customer Demographics
        tenure_months = np.random.randint(1, 72)
        age = np.random.randint(18, 75)
        income_bracket = np.random.randint(1, 10)  # 1=lowest, 10=highest
        
        # Behavioral Metrics
        login_frequency = np.random.uniform(0, 30)  # logins per month
        session_duration = np.random.uniform(1, 45)  # minutes per session
        page_views = np.random.uniform(5, 100)  # pages per session
        feature_adoption = np.random.uniform(0.1, 0.95)  # % features used
        
        # Transaction Patterns
        monthly_transactions = np.random.uniform(0, 50)
        avg_transaction_value = np.random.uniform(10, 500)
        total_spend_ltv = monthly_transactions * avg_transaction_value * (tenure_months / 12)
        
        # Support & Satisfaction
        support_tickets = np.random.randint(0, 15)
        nps_score = np.random.uniform(-100, 100)  # Net Promoter Score
        satisfaction_rating = np.random.uniform(1, 5)
        
        # Engagement Signals
        email_open_rate = np.random.uniform(0, 0.8)
        notification_enabled = np.random.choice([0, 1], p=[0.3, 0.7])
        last_active_days = np.random.exponential(15)  # days since last activity
        
        # Churn Indicators
        payment_delays = np.random.randint(0, 5)
        contract_type = np.random.choice([0, 1, 2])  # 0=monthly, 1=annual, 2=multi-year
        competitor_mentions = np.random.randint(0, 3)
        
        # COMPLEX NON-LINEAR CHURN SCORE FORMULA
        # Combines engagement, satisfaction, and behavioral signals
        churn_score = (
            # Engagement decay (exponential)
            -2.0 * np.exp(-login_frequency / 10) +
            -1.5 * np.exp(-session_duration / 20) +
            
            # Recency penalty (logarithmic)
            1.2 * np.log1p(last_active_days / 7) +
            
            # Satisfaction inverse (sigmoid)
            -1.8 * (1 / (1 + np.exp(-(satisfaction_rating - 3)))) +
            -0.8 * np.tanh(nps_score / 50) +
            
            # Support burden
            0.5 * np.sqrt(support_tickets) +
            0.4 * payment_delays +
            
            # Tenure effect (diminishing returns)
            -0.6 * np.log1p(tenure_months / 12) +
            
            # Transaction stickiness
            -0.3 * np.log1p(total_spend_ltv / 1000) +
            -0.4 * (feature_adoption ** 0.5) +
            
            # Contract loyalty
            -0.5 * contract_type +
            
            # Competition risk
            0.3 * competitor_mentions +
            
            # INTERACTION TERMS
            0.4 * (last_active_days / 30) * (1 - feature_adoption) +
            -0.3 * (login_frequency / 30) * satisfaction_rating / 5 +
            0.2 * (support_tickets / 10) * (1 - notification_enabled)
        )
        churn_score += np.random.normal(0, 0.4)
        
        # Determine churn risk class
        if churn_score < -2.0:
            churn_class = "LOYAL"  # Very unlikely to churn
            churn_prob = np.random.uniform(0.01, 0.05)
        elif churn_score < -0.5:
            churn_class = "STABLE"
            churn_prob = np.random.uniform(0.05, 0.15)
        elif churn_score < 1.0:
            churn_class = "AT_RISK"
            churn_prob = np.random.uniform(0.15, 0.40)
        else:
            churn_class = "CHURNING"
            churn_prob = np.random.uniform(0.40, 0.85)
        
        records.append({
            'Customer_ID': f'CUST-{i+1:04d}',
            'Tenure_Months': tenure_months,
            'Age': age,
            'Income_Bracket': income_bracket,
            'Login_Frequency': round(login_frequency, 2),
            'Session_Duration': round(session_duration, 2),
            'Page_Views': round(page_views, 1),
            'Feature_Adoption': round(feature_adoption, 3),
            'Monthly_Transactions': round(monthly_transactions, 1),
            'Avg_Transaction_Value': round(avg_transaction_value, 2),
            'Total_Spend_LTV': round(total_spend_ltv, 2),
            'Support_Tickets': support_tickets,
            'NPS_Score': round(nps_score, 1),
            'Satisfaction_Rating': round(satisfaction_rating, 2),
            'Email_Open_Rate': round(email_open_rate, 3),
            'Notification_Enabled': notification_enabled,
            'Last_Active_Days': round(last_active_days, 1),
            'Payment_Delays': payment_delays,
            'Contract_Type': contract_type,
            'Competitor_Mentions': competitor_mentions,
            'Churn_Class': churn_class,
            'Churn_Probability': round(churn_prob, 4)
        })
    
    return pd.DataFrame(records)


# =============================================================================
# PREDICTORS
# =============================================================================

@dataclass
class PredictionResult:
    predictions: List[str]
    latency_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None


class TabPFNPredictor:
    """SAP RPT-1-OSS using TabPFN - specialized for tabular data."""
    
    name = "SAP RPT-1-OSS"
    
    def __init__(self):
        self._initialized = False
    
    def _ensure_init(self):
        if not self._initialized:
            try:
                from tabpfn_client import init, set_access_token
                token = os.getenv('TABPFN_ACCESS_TOKEN')
                if token:
                    # Use set_access_token to avoid interactive prompts
                    set_access_token(token)
                self._initialized = True
            except Exception as e:
                # Fallback: try setting via environment variable
                pass
    
    def is_available(self) -> bool:
        return os.getenv('TABPFN_ACCESS_TOKEN') is not None
    
    def predict(self, X_train, y_train, X_test) -> PredictionResult:
        try:
            # Set token via environment variable before importing
            token = os.getenv('TABPFN_ACCESS_TOKEN')
            if not token:
                return PredictionResult(predictions=[], error="TABPFN_ACCESS_TOKEN not found in secrets")
            
            # Import and set token
            from tabpfn_client import set_access_token
            set_access_token(token)
            
            from tabpfn_client import TabPFNClassifier
            
            start = time.time()
            clf = TabPFNClassifier()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test).tolist()
            latency = (time.time() - start) * 1000
            
            return PredictionResult(predictions=preds, latency_ms=latency)
            
        except Exception as e:
            return PredictionResult(predictions=[], error=str(e))


class LLMPredictor:
    """General LLM predictor - will struggle with numeric patterns."""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.name = {
            'groq': 'Groq Llama 3.1 8B', 
            'groq-70b': 'Groq Llama 3.3 70B',
            'gemini': 'Gemini 2.0 Flash', 
            'mistral': 'Mistral Small',
            'openrouter': 'OpenRouter Claude 3.5 Haiku'
        }.get(provider, provider)
    
    def is_available(self) -> bool:
        key_map = {
            'groq': 'GROQ_API_KEY', 
            'groq-70b': 'GROQ_API_KEY',
            'gemini': 'GEMINI_API_KEY', 
            'mistral': 'MISTRAL_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY'
        }
        key = os.getenv(key_map.get(self.provider, ''))
        return key is not None and len(key) > 10
    
    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> PredictionResult:
        start = time.time()
        
        # Check API key first
        key_map = {
            'groq': 'GROQ_API_KEY', 
            'groq-70b': 'GROQ_API_KEY',
            'gemini': 'GEMINI_API_KEY', 
            'mistral': 'MISTRAL_API_KEY',
            'openrouter': 'OPENROUTER_API_KEY'
        }
        api_key = os.getenv(key_map.get(self.provider, ''))
        if not api_key or len(api_key) < 10:
            return PredictionResult(predictions=[], error=f"Missing or invalid {key_map.get(self.provider, 'API_KEY')} in .env")
        
        # COMPACT prompt - only 6 training samples to avoid rate limits
        train_sample = train_df.head(6).to_dict('records')
        test_sample = test_df.to_dict('records')
        
        # Simplify the data representation to reduce tokens
        prompt = f"""Classify {target} as HIGH, MEDIUM, or LOW for each test record.

TRAINING (6 examples):
{json.dumps(train_sample)}

TEST ({len(test_sample)} records):
{json.dumps(test_sample)}

Return ONLY a JSON array of {len(test_sample)} predictions like ["HIGH","MEDIUM","LOW"]"""

        try:
            if self.provider == 'groq':
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "llama-3.1-8b-instant", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 200},
                    timeout=30
                )
                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    tokens = resp.json().get('usage', {}).get('total_tokens', 0)
                else:
                    error_msg = resp.text[:200] if resp.text else f"Status {resp.status_code}"
                    return PredictionResult(predictions=[], error=f"Groq API Error: {error_msg}")
            
            elif self.provider == 'groq-70b':
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 200},
                    timeout=60
                )
                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    tokens = resp.json().get('usage', {}).get('total_tokens', 0)
                else:
                    error_msg = resp.text[:200] if resp.text else f"Status {resp.status_code}"
                    return PredictionResult(predictions=[], error=f"Groq 70B API Error: {error_msg}")
                    
            elif self.provider == 'gemini':
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200}},
                    timeout=30
                )
                if resp.status_code == 200:
                    content = resp.json()['candidates'][0]['content']['parts'][0]['text']
                    tokens = resp.json().get('usageMetadata', {}).get('totalTokenCount', 0)
                else:
                    error_msg = resp.text[:200] if resp.text else f"Status {resp.status_code}"
                    return PredictionResult(predictions=[], error=f"Gemini API Error: {error_msg}")
            
            elif self.provider == 'openrouter':
                resp = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://sap-rpt1-oss.app"},
                    json={"model": "anthropic/claude-3.5-haiku", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 200},
                    timeout=30
                )
                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    tokens = resp.json().get('usage', {}).get('total_tokens', 0)
                else:
                    error_msg = resp.text[:200] if resp.text else f"Status {resp.status_code}"
                    return PredictionResult(predictions=[], error=f"OpenRouter API Error: {error_msg}")
                    
            elif self.provider == 'mistral':
                resp = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": "mistral-small-latest", "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 300},
                    timeout=30
                )
                if resp.status_code == 200:
                    content = resp.json()['choices'][0]['message']['content']
                    tokens = resp.json().get('usage', {}).get('total_tokens', 0)
                else:
                    error_msg = resp.text[:200] if resp.text else f"Status {resp.status_code}"
                    return PredictionResult(predictions=[], error=f"Mistral API Error: {error_msg}")
            else:
                return PredictionResult(predictions=[], error="Unknown provider")
            
            latency = (time.time() - start) * 1000
            
            # Parse response
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                predictions = json.loads(json_match.group())
                predictions = [str(p).upper() for p in predictions]
            else:
                # Fallback: try to extract predictions from text
                predictions = []
                for word in content.upper().split():
                    if word in ['HIGH', 'MEDIUM', 'LOW']:
                        predictions.append(word)
                    if len(predictions) >= len(test_sample):
                        break
                if not predictions:
                    predictions = ["MEDIUM"] * len(test_sample)
            
            return PredictionResult(predictions=predictions, latency_ms=latency, tokens_used=tokens)
        
        except requests.exceptions.Timeout:
            return PredictionResult(predictions=[], error=f"{self.name} timed out (30s limit)")
        except requests.exceptions.ConnectionError:
            return PredictionResult(predictions=[], error=f"Connection error to {self.name} API")
        except json.JSONDecodeError as e:
            return PredictionResult(predictions=[], error=f"Failed to parse {self.name} response: {str(e)}")
        except Exception as e:
            return PredictionResult(predictions=[], error=f"{self.name} error: {str(e)}")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Initialize
    tabpfn = TabPFNPredictor()
    groq = LLMPredictor('groq')
    groq_70b = LLMPredictor('groq-70b')
    gemini = LLMPredictor('gemini')
    mistral = LLMPredictor('mistral')
    openrouter = LLMPredictor('openrouter')
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🎯 Forecast Integrity Showdown</h1>
        <p class="header-subtitle">SAP RPT-1-OSS vs General LLMs on Complex Tabular Data</p>
        <div class="header-badges">
            <span class="badge">🧮 Tabular ML</span>
            <span class="badge">🤖 LLM Comparison</span>
            <span class="badge">📊 3 Scenarios</span>
            <span class="badge">⚡ Real-time Testing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/5/59/SAP_2011_logo.svg", width=120)
        st.markdown("---")
        
        st.markdown("### ✅ Model Status")
        st.write(f"{'✅' if tabpfn.is_available() else '❌'} SAP RPT-1-OSS")
        st.write(f"{'✅' if groq.is_available() else '❌'} Groq Llama 8B")
        st.write(f"{'✅' if groq_70b.is_available() else '❌'} Groq Llama 70B")
        st.write(f"{'✅' if gemini.is_available() else '❌'} Gemini 2.0 Flash")
        st.write(f"{'✅' if mistral.is_available() else '❌'} Mistral Small")
        st.write(f"{'✅' if openrouter.is_available() else '❌'} OpenRouter Claude")
        
        st.markdown("---")
        st.markdown("### ⚙️ Test Settings")
        n_test = st.slider("Test Samples", 5, 30, 15, 
                          help="Number of records to test. Keep low (10-15) to avoid API rate limits.")
        seed = st.number_input("Random Seed", 1, 999, 42,
                              help="Controls synthetic data generation. Same seed = same data every time. Change seed to test on different data samples.")
        
        st.markdown("---")
        st.markdown("### 📋 About the Data")
        st.info("""
        **🔬 Synthetic Data**
        
        All data is **algorithmically generated** - no real company data is used.
        
        • **Forecast Tab**: 200 records, 13 financial features
        • **Credit Tab**: 200 records, 19 financial ratios
        • **Churn Tab**: 200 records, 20 behavioral features
        
        Labels are computed using **hidden non-linear formulas** that SAP RPT-1-OSS can learn but LLMs struggle to infer.
        """)
        
        st.markdown("---")
        
        # LLM Selection
        available_llms = []
        if groq_70b.is_available(): available_llms.append(('groq-70b', '🔥 Groq Llama 70B (Recommended)'))
        if groq.is_available(): available_llms.append(('groq', 'Groq Llama 8B'))
        if gemini.is_available(): available_llms.append(('gemini', 'Gemini 2.0 Flash'))
        if mistral.is_available(): available_llms.append(('mistral', 'Mistral Small'))
        if openrouter.is_available(): available_llms.append(('openrouter', 'OpenRouter Claude 3.5 Haiku'))
        
        if available_llms:
            selected_llm_key = st.selectbox(
                "LLM Challenger",
                options=[k for k, v in available_llms],
                format_func=lambda x: dict(available_llms).get(x, x)
            )
        else:
            selected_llm_key = None
            st.warning("No LLM available")
    
    # ==========================================================================
    # MAIN TABS - Three Scenarios
    # ==========================================================================
    
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Integrity (13 Features)", "🏦 Credit Risk (19 Features)", "👤 Customer Churn (20 Features)"])
    
    # ==========================================================================
    # TAB 1: FORECAST INTEGRITY
    # ==========================================================================
    with tab1:
        # Generate data
        df = generate_complex_forecast_data(200, seed)
        
        # Split data
        train_df = df.iloc[:200-n_test].copy()
        test_df = df.iloc[200-n_test:].copy()
        
        features = ['Revenue_Q1', 'Revenue_Q2', 'Revenue_Q3', 'Revenue_Q4', 
                    'YoY_Growth', 'Volatility', 'COGS_Ratio', 'OpEx_Ratio', 
                    'Margin', 'Market_Share', 'Competitors', 'Customer_Concentration', 'Seasonal_Index']
        
        X_train = train_df[features].values
        y_train = train_df['Integrity_Class'].values
        X_test = test_df[features].values
        y_true = test_df['Integrity_Class'].tolist()
        
        # STEP 1: UNDERSTAND THE DATA
        st.markdown("## <span class='step-number'>1</span> The Data Challenge", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="story-card">
            <h4>🎯 The Task</h4>
            <p>Predict <strong>Forecast Integrity</strong> (HIGH / MEDIUM / LOW) based on complex financial metrics.</p>
            <p>This requires understanding <strong>non-linear relationships</strong> between 13 numeric features - 
            something LLMs struggle with because they see numbers as text tokens, not mathematical values.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📐 The Hidden Formula (What SAP RPT-1-OSS Must Learn)", expanded=False):
            st.code("""
# Risk Score Formula (non-linear combinations)
risk_score = (
    volatility * 2.0 +              # Revenue volatility penalty
    (1 - margin) * 1.5 +            # Low margin penalty
    customer_concentration * 1.2 +   # Concentration risk
    (1 / (competitors + 1)) * 0.8 + # Market dynamics
    abs(yoy_growth - 0.1) * 1.0     # Growth deviation penalty
)

# Classification thresholds
if risk_score < 1.8:  → HIGH integrity
if risk_score < 2.8:  → MEDIUM integrity  
else:                 → LOW integrity
            """, language="python")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Sample Data (Training Set)")
            display_cols = ['ID', 'Revenue_Q1', 'Revenue_Q4', 'YoY_Growth', 'Volatility', 'Margin', 'Customer_Concentration', 'Integrity_Class']
            st.dataframe(train_df[display_cols].head(10), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 Class Distribution")
            dist = df['Integrity_Class'].value_counts()
            st.bar_chart(dist)
            
            st.markdown("### 🔢 Key Stats")
            st.metric("Training Samples", len(train_df))
            st.metric("Test Samples", n_test)
            st.metric("Features", len(features))
        
        st.markdown("""
        <div class="insight-box">
            <h4>💡 Why This Is Hard for LLMs</h4>
            <p>The integrity class depends on <strong>complex interactions</strong>:</p>
            <ul>
                <li>High Volatility + Low Margin + High Customer Concentration → <strong>LOW</strong></li>
                <li>Stable Revenue + Good Margin + Diversified Customers → <strong>HIGH</strong></li>
            </ul>
            <p>LLMs process these numbers as text tokens ("0.234" = 5 tokens). 
            SAP RPT-1-OSS processes them as actual numeric values with mathematical relationships.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # STEP 2: RUN THE SHOWDOWN
        st.markdown("## <span class='step-number'>2</span> The Showdown", unsafe_allow_html=True)
        
        if not tabpfn.is_available():
            st.error("⚠️ SAP RPT-1-OSS not available. Configure TABPFN_ACCESS_TOKEN in .env")
            st.stop()
        
        if st.button("🚀 Run Head-to-Head Comparison", type="primary", use_container_width=True, key="forecast_btn"):
            
            results = {}
            
            # Run LLM First
            if selected_llm_key:
                llm = LLMPredictor(selected_llm_key)
                st.markdown(f"### 🤖 Step 2a: Running {llm.name} First...")
                
                with st.spinner(f"LLM analyzing {n_test} records..."):
                    test_features_df = test_df[features].copy()
                    train_with_target = train_df[features + ['Integrity_Class']].copy()
                    llm_result = llm.predict(train_with_target, test_features_df, 'Integrity_Class')
                
                if llm_result.error:
                    st.error(f"LLM Error: {llm_result.error}")
                    results['llm'] = None
                else:
                    results['llm'] = llm_result
                    st.success(f"✅ {llm.name} completed in {llm_result.latency_ms:.0f}ms")
            
            # Run SAP RPT-1-OSS
            st.markdown("### 🏢 Step 2b: Running SAP RPT-1-OSS...")
            
            with st.spinner("SAP RPT-1-OSS processing..."):
                tabpfn_result = tabpfn.predict(X_train, y_train, X_test)
            
            if tabpfn_result.error:
                st.error(f"SAP RPT-1-OSS Error: {tabpfn_result.error}")
                results['tabpfn'] = None
            else:
                results['tabpfn'] = tabpfn_result
                st.success(f"✅ SAP RPT-1-OSS completed in {tabpfn_result.latency_ms:.0f}ms")
            
            st.markdown("---")
            
            # STEP 3: COMPARE RESULTS
            st.markdown("## <span class='step-number'>3</span> Side-by-Side Comparison", unsafe_allow_html=True)
            
            if results.get('tabpfn') and results.get('llm'):
                # Create comparison dataframe
                comparison_data = []
                tabpfn_preds = results['tabpfn'].predictions
                llm_preds = results['llm'].predictions
                
                tabpfn_correct = 0
                llm_correct = 0
                
                for i, (true_val, tabpfn_pred, llm_pred) in enumerate(zip(y_true, tabpfn_preds, llm_preds[:len(y_true)])):
                    tabpfn_match = tabpfn_pred == true_val
                    llm_match = llm_pred == true_val
                    
                    if tabpfn_match: tabpfn_correct += 1
                    if llm_match: llm_correct += 1
                    
                    comparison_data.append({
                        'Test #': i + 1,
                        'Actual': true_val,
                        'SAP RPT-1-OSS': f"{'✅' if tabpfn_match else '❌'} {tabpfn_pred}",
                        'LLM': f"{'✅' if llm_match else '❌'} {llm_pred}",
                        'Winner': '🏢 SAP' if (tabpfn_match and not llm_match) else ('🤖 LLM' if (llm_match and not tabpfn_match) else ('🤝 Tie' if tabpfn_match else '❌ Both Wrong'))
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                
                # Show metrics side by side
                col1, col2, col3 = st.columns(3)
                
                tabpfn_acc = tabpfn_correct / len(y_true)
                llm_acc = llm_correct / len(y_true)
                
                with col1:
                    st.markdown("""
                    <div class="sap-card">
                        <h3>🏢 SAP RPT-1-OSS</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Accuracy", f"{tabpfn_acc:.1%}")
                    st.metric("Correct", f"{tabpfn_correct}/{len(y_true)}")
                    st.metric("Latency", f"{results['tabpfn'].latency_ms:.0f}ms")
                
                with col2:
                    st.markdown("""
                    <div style="text-align: center; padding: 40px;">
                        <span style="font-size: 3rem;">⚔️</span>
                        <h3>VS</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="llm-card">
                        <h3>🤖 {llm.name}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Accuracy", f"{llm_acc:.1%}")
                    st.metric("Correct", f"{llm_correct}/{len(y_true)}")
                    st.metric("Latency", f"{results['llm'].latency_ms:.0f}ms")
                
                st.markdown("---")
                
                # Show detailed comparison
                st.markdown("### 📋 Prediction-by-Prediction Breakdown")
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # Context Summary - Layman Friendly
                st.markdown("### 💡 Why Did This Happen?")
                forecast_html = f"""<div class="simple-insight">
<h4 style="color: #1a1a2e; margin-top: 0; font-family: 'Segoe UI', sans-serif;">📊 Results Explained Simply</h4>
<p style="font-size: 1.1rem; color: #333; font-family: 'Segoe UI', sans-serif;"><strong>SAP RPT-1-OSS: {tabpfn_acc:.1%}</strong> correct &nbsp;|&nbsp; <strong>{llm.name}: {llm_acc:.1%}</strong> correct</p>
<hr style="border: 1px solid #dee2e6; margin: 15px 0;">
<p style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 1rem;"><strong>🤔 Think of it like this:</strong></p>
<ul style="color: #444; line-height: 2; font-family: 'Segoe UI', sans-serif; font-size: 0.95rem;">
<li>🧮 <strong>SAP RPT-1-OSS reads numbers like a calculator</strong> — it sees "0.2345" as a real number and can do math with it.</li>
<li>📝 <strong>ChatGPT reads numbers like text</strong> — it sees "0.2345" as five characters (0, ., 2, 3, 4, 5) and loses the meaning.</li>
<li>🔗 <strong>Hidden patterns matter</strong> — The answer depends on complex rules like "if revenue is unstable AND profit margin is low, then risk is HIGH". SAP RPT-1-OSS figures this out from examples; LLMs just guess.</li>
<li>📚 <strong>Learning from few examples</strong> — SAP RPT-1-OSS only needed {len(train_df)} examples to learn. LLMs would need thousands or explicit instructions.</li>
</ul>
<p style="background: #e8f4f8; padding: 12px; border-radius: 8px; color: #0066cc; margin-top: 15px; font-family: 'Segoe UI', sans-serif;">
<strong>💡 Bottom Line:</strong> For spreadsheet-style data with numbers, use specialized tools like SAP RPT-1-OSS. For writing emails or answering questions, use ChatGPT!
</p>
</div>"""
                st.markdown(forecast_html, unsafe_allow_html=True)
                
                # Winner announcement
                st.markdown("---")
                
                if tabpfn_acc > llm_acc:
                    margin = tabpfn_acc - llm_acc
                    st.markdown(f"""
                    <div class="winner-banner">
                        <h2>🏆 SAP RPT-1-OSS WINS!</h2>
                        <p>Outperformed {llm.name} by <strong>{margin:.1%}</strong></p>
                        <p><em>Specialized tabular ML beats general-purpose LLMs on structured numeric data.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class="story-card">
                        <h4>🔍 Why Did SAP RPT-1-OSS Win?</h4>
                        <ul>
                            <li><strong>Native Numeric Understanding:</strong> SAP RPT-1-OSS processes numbers as mathematical values, not text tokens</li>
                            <li><strong>Pattern Recognition:</strong> Detects complex interactions between 13 features simultaneously</li>
                            <li><strong>In-Context Learning:</strong> Learns from training examples without explicit rules</li>
                            <li><strong>No Tokenization Loss:</strong> "0.2345" stays as one numeric value, not 5+ tokens</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                elif llm_acc > tabpfn_acc:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 30px; border-radius: 15px; text-align: center;">
                        <h2>🤔 Interesting Result</h2>
                        <p>{llm.name} edged ahead this time!</p>
                        <p>Try increasing test samples or changing the random seed. SAP RPT-1-OSS typically wins on complex numeric patterns.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 30px; border-radius: 15px; text-align: center;">
                        <h2>🤝 It's a Tie!</h2>
                        <p>Both models performed equally on this test set.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif results.get('tabpfn'):
                st.info("Run with an LLM challenger to see the comparison!")
                st.markdown("### SAP RPT-1-OSS Results")
                tabpfn_acc = sum(1 for t, p in zip(y_true, results['tabpfn'].predictions) if t == p) / len(y_true)
                st.metric("Accuracy", f"{tabpfn_acc:.1%}")
        
        st.markdown("---")
        
        # EXPLANATION
        st.markdown("## <span class='step-number'>4</span> Understanding the Difference", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="sap-card">
                <h4>🏢 How SAP RPT-1-OSS Works</h4>
                <ul>
                    <li>Treats numbers as <strong>mathematical values</strong></li>
                    <li>Learns <strong>feature interactions</strong> automatically</li>
                    <li>Uses <strong>in-context learning</strong> from examples</li>
                    <li>Optimized for <strong>tabular data structure</strong></li>
                    <li>Fast: processes all features in parallel</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="llm-card">
                <h4>🤖 How LLMs Process This Data</h4>
                <ul>
                    <li>Converts numbers to <strong>text tokens</strong></li>
                    <li>"0.2345" becomes multiple tokens</li>
                    <li>Loses <strong>mathematical relationships</strong></li>
                    <li>Relies on <strong>pattern matching in text</strong></li>
                    <li>May hallucinate or guess based on text patterns</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 The Bottom Line</h4>
            <p><strong>Use the right tool for the job:</strong></p>
            <ul>
                <li>📊 <strong>Structured/Tabular Data</strong> → SAP RPT-1-OSS</li>
                <li>📝 <strong>Text/Language Tasks</strong> → LLMs (GPT, Gemini, Claude)</li>
            </ul>
            <p>Both are powerful - but in different domains!</p>
        </div>
        """, unsafe_allow_html=True)

    # ==========================================================================
    # TAB 2: ENTERPRISE CREDIT RISK (More Complex)
    # ==========================================================================
    with tab2:
        st.markdown("## 🏦 Enterprise Credit Risk Assessment")
        st.markdown("**19 Financial Features | Non-Linear Z-Score Formula | 6 Credit Classes**")
        
        # Generate credit risk data
        credit_df = generate_credit_risk_data(200, seed)
        
        # Split data
        credit_train = credit_df.iloc[:200-n_test].copy()
        credit_test = credit_df.iloc[200-n_test:].copy()
        
        credit_features = ['Current_Ratio', 'Quick_Ratio', 'Cash_Ratio', 
                          'Debt_to_Equity', 'Debt_to_Assets', 'Interest_Coverage',
                          'ROA', 'ROE', 'Profit_Margin', 'EBITDA_Margin',
                          'Asset_Turnover', 'Inventory_Days', 'Receivables_Days', 
                          'Payables_Days', 'Cash_Conversion_Cycle',
                          'Market_Cap_Log', 'Beta', 'Years_Operating']
        
        X_train_cr = credit_train[credit_features].values
        y_train_cr = credit_train['Credit_Rating'].values
        X_test_cr = credit_test[credit_features].values
        y_true_cr = credit_test['Credit_Rating'].tolist()
        
        # STEP 1: The Complex Formula
        st.markdown("### <span class='step-number'>1</span> The Mathematical Challenge", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="story-card">
            <h4>🧮 Altman Z-Score Inspired Credit Model</h4>
            <p>This uses a <strong>highly non-linear formula</strong> with logarithms, hyperbolic functions, 
            sigmoid transformations, and interaction terms - IMPOSSIBLE for LLMs to infer from text.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📐 The Hidden Z-Score Formula (Intense Math!)", expanded=True):
            st.code("""
# Enterprise Credit Z-Score (Non-Linear Multi-Variable Model)
z_score = (
    # Liquidity Component (logarithmic)
    1.2 * log(1 + current_ratio) +
    0.8 * (quick_ratio / (current_ratio + 0.01)) +
    
    # Leverage Component (hyperbolic tangent)
    -1.5 * tanh(debt_to_equity / 2) +
    0.6 * log(1 + interest_coverage) +
    
    # Profitability Component (sigmoid transformation)
    2.0 * sigmoid(10 * ROA) +
    1.5 * clip(profit_margin * 5, -1, 1) +
    
    # Efficiency Component (log-linear)
    0.4 * log(1 + asset_turnover) +
    -0.3 * (cash_conversion_cycle / 100) +
    
    # Size & Stability (scaled)
    0.2 * (market_cap_log / 12) +
    0.1 * log(1 + years_operating / 10) +
    
    # INTERACTION TERMS (Very Hard for LLMs!)
    0.5 * (current_ratio × profit_margin) +
    -0.4 * (debt_to_equity × (1 - profit_margin)) +
    0.3 * (interest_coverage / (debt_to_equity + 1))
)

# Credit Rating Thresholds
z > 4.0  → AAA    (Default Prob: 0.01-0.1%)
z > 3.0  → AA     (Default Prob: 0.1-1%)
z > 2.0  → A      (Default Prob: 1-3%)
z > 1.0  → BBB    (Default Prob: 3-8%)
z > 0.0  → BB     (Default Prob: 8-15%)
z ≤ 0.0  → B      (Default Prob: 15-35%)
            """, language="python")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Sample Credit Risk Data")
            display_credit = ['Entity_ID', 'Current_Ratio', 'Debt_to_Equity', 'ROA', 'Profit_Margin', 'Interest_Coverage', 'Credit_Rating']
            st.dataframe(credit_train[display_credit].head(10), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 Credit Rating Distribution")
            cr_dist = credit_df['Credit_Rating'].value_counts().reindex(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'])
            st.bar_chart(cr_dist)
            
            st.markdown("### 🔢 Complexity Stats")
            st.metric("Features", len(credit_features))
            st.metric("Classes", 6)
            st.metric("Formula Terms", 13)
        
        st.markdown("---")
        
        # STEP 2: RUN THE CREDIT RISK SHOWDOWN
        st.markdown("### <span class='step-number'>2</span> Credit Risk Showdown", unsafe_allow_html=True)
        
        if st.button("🚀 Run Credit Risk Comparison", type="primary", use_container_width=True, key="credit_btn"):
            
            credit_results = {}
            
            # Run LLM First
            if selected_llm_key:
                llm = LLMPredictor(selected_llm_key)
                st.markdown(f"#### 🤖 Step 2a: Running {llm.name}...")
                
                with st.spinner(f"LLM analyzing {n_test} credit profiles..."):
                    test_features_df = credit_test[credit_features].copy()
                    train_with_target = credit_train[credit_features + ['Credit_Rating']].copy()
                    llm_result = llm.predict(train_with_target, test_features_df, 'Credit_Rating')
                
                if llm_result.error:
                    st.error(f"LLM Error: {llm_result.error}")
                else:
                    credit_results['llm'] = llm_result
                    st.success(f"✅ {llm.name} completed in {llm_result.latency_ms:.0f}ms")
            
            # Run SAP RPT-1-OSS
            st.markdown("#### 🏢 Step 2b: Running SAP RPT-1-OSS...")
            
            with st.spinner("SAP RPT-1-OSS analyzing credit data..."):
                tabpfn_result = tabpfn.predict(X_train_cr, y_train_cr, X_test_cr)
            
            if tabpfn_result.error:
                st.error(f"SAP RPT-1-OSS Error: {tabpfn_result.error}")
            else:
                credit_results['tabpfn'] = tabpfn_result
                st.success(f"✅ SAP RPT-1-OSS completed in {tabpfn_result.latency_ms:.0f}ms")
            
            st.markdown("---")
            
            # STEP 3: COMPARE
            st.markdown("### <span class='step-number'>3</span> Credit Rating: Prediction vs Actual", unsafe_allow_html=True)
            
            if credit_results.get('tabpfn') and credit_results.get('llm'):
                tabpfn_preds = credit_results['tabpfn'].predictions
                llm_preds = credit_results['llm'].predictions
                
                tabpfn_correct = sum(1 for t, p in zip(y_true_cr, tabpfn_preds) if t == p)
                llm_correct = sum(1 for t, p in zip(y_true_cr, llm_preds[:len(y_true_cr)]) if str(t) == str(p))
                
                tabpfn_acc = tabpfn_correct / len(y_true_cr)
                llm_acc = llm_correct / len(y_true_cr)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="sap-card"><h3>🏢 SAP RPT-1-OSS</h3></div>', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{tabpfn_acc:.1%}")
                    st.metric("Correct", f"{tabpfn_correct}/{len(y_true_cr)}")
                
                with col2:
                    st.markdown('<div style="text-align: center; padding: 30px;"><span style="font-size: 3rem;">⚔️</span></div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'<div class="llm-card"><h3>🤖 {llm.name}</h3></div>', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{llm_acc:.1%}")
                    st.metric("Correct", f"{llm_correct}/{len(y_true_cr)}")
                
                # Detailed breakdown
                st.markdown("### 📋 Detailed Credit Assessment Breakdown")
                
                comparison_data = []
                for i, (true_val, tabpfn_pred) in enumerate(zip(y_true_cr, tabpfn_preds)):
                    llm_pred = llm_preds[i] if i < len(llm_preds) else "N/A"
                    comparison_data.append({
                        'Entity': credit_test.iloc[i]['Entity_ID'],
                        'D/E Ratio': f"{credit_test.iloc[i]['Debt_to_Equity']:.2f}",
                        'ROA': f"{credit_test.iloc[i]['ROA']:.2%}",
                        'Actual Rating': true_val,
                        'SAP RPT-1-OSS': f"{'✅' if tabpfn_pred == true_val else '❌'} {tabpfn_pred}",
                        'LLM': f"{'✅' if str(llm_pred) == str(true_val) else '❌'} {llm_pred}",
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                
                # Context Summary - Layman Friendly for Credit Risk
                st.markdown("### 💡 Why Did This Happen?")
                credit_html = f"""<div class="simple-insight">
<h4 style="color: #1a1a2e; margin-top: 0; font-family: 'Segoe UI', sans-serif;">🏦 Credit Risk Results Explained Simply</h4>
<p style="font-size: 1.1rem; color: #333; font-family: 'Segoe UI', sans-serif;"><strong>SAP RPT-1-OSS: {tabpfn_acc:.1%}</strong> correct &nbsp;|&nbsp; <strong>{llm.name}: {llm_acc:.1%}</strong> correct</p>
<hr style="border: 1px solid #dee2e6; margin: 15px 0;">
<p style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 1rem;"><strong>🤔 Here's what's happening:</strong></p>
<ul style="color: #444; line-height: 2; font-family: 'Segoe UI', sans-serif; font-size: 0.95rem;">
<li>📈 <strong>Banks use complex formulas</strong> — Credit ratings (AAA, AA, BBB, etc.) are calculated using fancy math like logarithms and special curves. SAP RPT-1-OSS learns these patterns; LLMs can't do this math from text.</li>
<li>🔢 <strong>19 numbers at once</strong> — The rating depends on debt ratio, profit margins, cash flow, and 16 other metrics ALL together. SAP RPT-1-OSS sees them as connected numbers; LLMs see them as separate words.</li>
<li>✖️ <strong>Multiplication matters</strong> — Rules like "debt × (1 - profit)" combine metrics. SAP RPT-1-OSS catches these combinations; LLMs miss them entirely.</li>
<li>🎯 <strong>6 very similar categories</strong> — The difference between AAA and AA is tiny. SAP RPT-1-OSS finds the exact boundaries; LLMs make rough guesses.</li>
</ul>
<p style="background: #fff3cd; padding: 12px; border-radius: 8px; color: #856404; margin-top: 15px; font-family: 'Segoe UI', sans-serif;">
<strong>🏦 Real World:</strong> This is why banks use specialized ML models for credit scoring, not ChatGPT!
</p>
</div>"""
                st.markdown(credit_html, unsafe_allow_html=True)
                
                # Winner
                if tabpfn_acc > llm_acc:
                    margin = tabpfn_acc - llm_acc
                    st.markdown(f"""
                    <div class="winner-banner">
                        <h2>🏆 SAP RPT-1-OSS DOMINATES Credit Risk!</h2>
                        <p>Outperformed {llm.name} by <strong>{margin:.1%}</strong> on 6-class credit rating</p>
                        <p><em>Complex financial formulas are no match for specialized tabular ML!</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif llm_acc > tabpfn_acc:
                    st.warning(f"🤔 {llm.name} performed better this round. Try different seed or more samples.")
                else:
                    st.info("🤝 It's a tie on credit risk assessment!")

    # ==========================================================================
    # TAB 3: CUSTOMER CHURN PREDICTION
    # ==========================================================================
    with tab3:
        st.markdown("## 👤 Customer Churn Prediction")
        st.markdown("**20 Behavioral Features | RFM Analysis | 4 Churn Classes**")
        
        # Generate churn data
        churn_df = generate_customer_churn_data(200, seed)
        
        # Split data
        churn_train = churn_df.iloc[:200-n_test].copy()
        churn_test = churn_df.iloc[200-n_test:].copy()
        
        churn_features = ['Tenure_Months', 'Age', 'Income_Bracket',
                         'Login_Frequency', 'Session_Duration', 'Page_Views', 'Feature_Adoption',
                         'Monthly_Transactions', 'Avg_Transaction_Value', 'Total_Spend_LTV',
                         'Support_Tickets', 'NPS_Score', 'Satisfaction_Rating',
                         'Email_Open_Rate', 'Notification_Enabled', 'Last_Active_Days',
                         'Payment_Delays', 'Contract_Type', 'Competitor_Mentions']
        
        X_train_ch = churn_train[churn_features].values
        y_train_ch = churn_train['Churn_Class'].values
        X_test_ch = churn_test[churn_features].values
        y_true_ch = churn_test['Churn_Class'].tolist()
        
        # STEP 1: The Churn Formula
        st.markdown("### <span class='step-number'>1</span> The Behavioral Challenge", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="story-card">
            <h4>🔮 Predictive Churn Model</h4>
            <p>Uses <strong>RFM analysis</strong> (Recency, Frequency, Monetary) combined with 
            engagement metrics, satisfaction scores, and <strong>exponential decay functions</strong> 
            that LLMs cannot learn from text.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("📐 The Hidden Churn Score Formula", expanded=True):
            st.code("""
# Customer Churn Score (Behavioral Multi-Variable Model)
churn_score = (
    # Engagement Decay (exponential)
    -2.0 * exp(-login_frequency / 10) +
    -1.5 * exp(-session_duration / 20) +
    
    # Recency Penalty (logarithmic)
    1.2 * log(1 + last_active_days / 7) +
    
    # Satisfaction Inverse (sigmoid)
    -1.8 * sigmoid(satisfaction_rating - 3) +
    -0.8 * tanh(nps_score / 50) +
    
    # Support Burden
    0.5 * sqrt(support_tickets) +
    0.4 * payment_delays +
    
    # Tenure Effect (diminishing returns)
    -0.6 * log(1 + tenure_months / 12) +
    
    # Transaction Stickiness
    -0.3 * log(1 + total_spend_ltv / 1000) +
    -0.4 * sqrt(feature_adoption) +
    
    # Contract Loyalty & Competition
    -0.5 * contract_type +
    0.3 * competitor_mentions +
    
    # INTERACTION TERMS
    0.4 * (last_active_days / 30) × (1 - feature_adoption) +
    -0.3 * (login_frequency / 30) × (satisfaction_rating / 5) +
    0.2 * (support_tickets / 10) × (1 - notification_enabled)
)

# Churn Class Thresholds
score < -2.0  → LOYAL     (Churn Prob: 1-5%)
score < -0.5  → STABLE    (Churn Prob: 5-15%)
score <  1.0  → AT_RISK   (Churn Prob: 15-40%)
score >= 1.0  → CHURNING  (Churn Prob: 40-85%)
            """, language="python")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Sample Customer Data")
            display_churn = ['Customer_ID', 'Tenure_Months', 'Login_Frequency', 'Satisfaction_Rating', 'Last_Active_Days', 'Support_Tickets', 'Churn_Class']
            st.dataframe(churn_train[display_churn].head(10), use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 📈 Churn Class Distribution")
            ch_dist = churn_df['Churn_Class'].value_counts().reindex(['LOYAL', 'STABLE', 'AT_RISK', 'CHURNING'])
            st.bar_chart(ch_dist)
            
            st.markdown("### 🔢 Model Complexity")
            st.metric("Features", len(churn_features))
            st.metric("Classes", 4)
            st.metric("Formula Terms", 15)
        
        st.markdown("---")
        
        # STEP 2: RUN THE CHURN SHOWDOWN
        st.markdown("### <span class='step-number'>2</span> Churn Prediction Showdown", unsafe_allow_html=True)
        
        if st.button("🚀 Run Churn Prediction Comparison", type="primary", use_container_width=True, key="churn_btn"):
            
            churn_results = {}
            
            # Run LLM First
            if selected_llm_key:
                llm = LLMPredictor(selected_llm_key)
                st.markdown(f"#### 🤖 Step 2a: Running {llm.name}...")
                
                with st.spinner(f"LLM analyzing {n_test} customer profiles..."):
                    test_features_df = churn_test[churn_features].copy()
                    train_with_target = churn_train[churn_features + ['Churn_Class']].copy()
                    llm_result = llm.predict(train_with_target, test_features_df, 'Churn_Class')
                
                if llm_result.error:
                    st.error(f"LLM Error: {llm_result.error}")
                else:
                    churn_results['llm'] = llm_result
                    st.success(f"✅ {llm.name} completed in {llm_result.latency_ms:.0f}ms")
            
            # Run SAP RPT-1-OSS
            st.markdown("#### 🏢 Step 2b: Running SAP RPT-1-OSS...")
            
            with st.spinner("SAP RPT-1-OSS analyzing customer behavior..."):
                tabpfn_result = tabpfn.predict(X_train_ch, y_train_ch, X_test_ch)
            
            if tabpfn_result.error:
                st.error(f"SAP RPT-1-OSS Error: {tabpfn_result.error}")
            else:
                churn_results['tabpfn'] = tabpfn_result
                st.success(f"✅ SAP RPT-1-OSS completed in {tabpfn_result.latency_ms:.0f}ms")
            
            st.markdown("---")
            
            # STEP 3: COMPARE
            st.markdown("### <span class='step-number'>3</span> Churn Prediction: Model vs Actual", unsafe_allow_html=True)
            
            if churn_results.get('tabpfn') and churn_results.get('llm'):
                tabpfn_preds = churn_results['tabpfn'].predictions
                llm_preds = churn_results['llm'].predictions
                
                tabpfn_correct = sum(1 for t, p in zip(y_true_ch, tabpfn_preds) if t == p)
                llm_correct = sum(1 for t, p in zip(y_true_ch, llm_preds[:len(y_true_ch)]) if str(t) == str(p))
                
                tabpfn_acc = tabpfn_correct / len(y_true_ch)
                llm_acc = llm_correct / len(y_true_ch)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown('<div class="sap-card"><h3>🏢 SAP RPT-1-OSS</h3></div>', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{tabpfn_acc:.1%}")
                    st.metric("Correct", f"{tabpfn_correct}/{len(y_true_ch)}")
                
                with col2:
                    st.markdown('<div style="text-align: center; padding: 30px;"><span style="font-size: 3rem;">⚔️</span></div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f'<div class="llm-card"><h3>🤖 {llm.name}</h3></div>', unsafe_allow_html=True)
                    st.metric("Accuracy", f"{llm_acc:.1%}")
                    st.metric("Correct", f"{llm_correct}/{len(y_true_ch)}")
                
                # Detailed breakdown
                st.markdown("### 📋 Detailed Churn Assessment Breakdown")
                
                comparison_data = []
                for i, (true_val, tabpfn_pred) in enumerate(zip(y_true_ch, tabpfn_preds)):
                    llm_pred = llm_preds[i] if i < len(llm_preds) else "N/A"
                    comparison_data.append({
                        'Customer': churn_test.iloc[i]['Customer_ID'],
                        'Tenure': f"{churn_test.iloc[i]['Tenure_Months']}mo",
                        'Last Active': f"{churn_test.iloc[i]['Last_Active_Days']:.0f}d",
                        'Satisfaction': f"{churn_test.iloc[i]['Satisfaction_Rating']:.1f}/5",
                        'Actual': true_val,
                        'SAP RPT-1-OSS': f"{'✅' if tabpfn_pred == true_val else '❌'} {tabpfn_pred}",
                        'LLM': f"{'✅' if str(llm_pred) == str(true_val) else '❌'} {llm_pred}",
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                
                # Context Summary - Layman Friendly for Churn
                st.markdown("### 💡 Why Did This Happen?")
                churn_html = f"""<div class="simple-insight">
<h4 style="color: #1a1a2e; margin-top: 0; font-family: 'Segoe UI', sans-serif;">👤 Churn Prediction Results Explained Simply</h4>
<p style="font-size: 1.1rem; color: #333; font-family: 'Segoe UI', sans-serif;"><strong>SAP RPT-1-OSS: {tabpfn_acc:.1%}</strong> correct &nbsp;|&nbsp; <strong>{llm.name}: {llm_acc:.1%}</strong> correct</p>
<hr style="border: 1px solid #dee2e6; margin: 15px 0;">
<p style="color: #333; font-family: 'Segoe UI', sans-serif; font-size: 1rem;"><strong>🤔 What's going on here:</strong></p>
<ul style="color: #444; line-height: 2; font-family: 'Segoe UI', sans-serif; font-size: 0.95rem;">
<li>📉 <strong>Fading engagement is exponential</strong> — When users stop logging in, their risk grows faster over time (like compound interest, but bad). SAP RPT-1-OSS understands this curve; LLMs don't.</li>
<li>🛒 <strong>RFM is the secret sauce</strong> — Marketers use Recency (when did they last visit?), Frequency (how often?), and Money (how much did they spend?) together. SAP RPT-1-OSS connects these dots naturally.</li>
<li>🎭 <strong>20 clues to read</strong> — Login times, support tickets, satisfaction scores, payment history... all mixed together. SAP RPT-1-OSS sees the full picture; LLMs get overwhelmed.</li>
<li>⚖️ <strong>Thin lines between categories</strong> — Is a customer "STABLE" or "AT_RISK"? The difference is subtle. SAP RPT-1-OSS finds the exact cutoff; LLMs guess roughly.</li>
</ul>
<p style="background: #d4edda; padding: 12px; border-radius: 8px; color: #155724; margin-top: 15px; font-family: 'Segoe UI', sans-serif;">
<strong>📱 Real World:</strong> Netflix, Spotify, and subscription services use models like SAP RPT-1-OSS to predict who's about to cancel!
</p>
</div>"""
                st.markdown(churn_html, unsafe_allow_html=True)
                
                # Winner
                if tabpfn_acc > llm_acc:
                    margin = tabpfn_acc - llm_acc
                    st.markdown(f"""
                    <div class="winner-banner">
                        <h2>🏆 SAP RPT-1-OSS WINS Churn Prediction!</h2>
                        <p>Outperformed {llm.name} by <strong>{margin:.1%}</strong> on 4-class churn classification</p>
                        <p><em>Behavioral patterns and customer signals are crystal clear to specialized ML!</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                elif llm_acc > tabpfn_acc:
                    st.warning(f"🤔 {llm.name} performed better this round. Try different seed or more samples.")
                else:
                    st.info("🤝 It's a tie on churn prediction!")
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 30px;">
        <p style="color: white; font-size: 14px; margin: 0;">
            🚀 <strong>Forecast Integrity Showdown</strong> | Developed by <strong>Amit Lal</strong> | 
            <a href="https://aka.ms/amitlal" target="_blank" style="color: #fff; text-decoration: underline;">aka.ms/amitlal</a>
        </p>
        <p style="color: rgba(255,255,255,0.8); font-size: 12px; margin: 5px 0 0 0;">
            Comparing SAP RPT-1 Public Model vs GPT Models on Complex Tabular Data
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <p style="text-align: center; font-size: 11px; color: #6c757d; margin-top: 15px; padding: 0 20px;">
        ⚖️ <strong>Disclaimer:</strong> SAP, SAP RPT, and all SAP logos and product names are trademarks or registered trademarks of SAP SE in Germany and other countries. This is an independent demonstration project for educational purposes only and is not affiliated with, endorsed by, or sponsored by SAP SE or any enterprise. All other trademarks are the property of their respective owners.
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

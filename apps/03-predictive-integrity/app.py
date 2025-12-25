import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
from typing import Dict, List, Tuple, Optional
from utils.failure_data_generator import generate_job_failure_data, generate_transport_failure_data, generate_interface_failure_data, detect_drift
from utils.sap_rpt1_client import SAPRPT1Client, SAPRPT1OSSClient

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="SAP Predictive Integrity | Operational Risk",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark/Light mode optimization
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
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #f093fb 0%, #f5576c 30%, #4facfe 60%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
    }
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        margin: 10px 0 0 0;
    }
    .badge {
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        backdrop-filter: blur(10px);
        margin: 0 5px;
    }
    .story-card {
        background: rgba(128, 128, 128, 0.1);
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #0066cc;
    }
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
    }
    .step-number {
        background: #0066cc;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00c851; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'token' not in st.session_state:
    st.session_state.token = ""
if 'token_validated' not in st.session_state:
    st.session_state.token_validated = False
if 'model_type' not in st.session_state:
    st.session_state.model_type = "SAP-RPT-1-OSS (Public)"
if 'hf_token' not in st.session_state:
    st.session_state.hf_token = ""
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'scenario' not in st.session_state:
    st.session_state.scenario = "Job Failure"
if 'seed' not in st.session_state:
    st.session_state.seed = 42
if 'drift_detected' not in st.session_state:
    st.session_state.drift_detected = False

# =============================================================================
# HELPERS
# =============================================================================

def get_remediation_playbook(scenario: str, risk_label: str, row: Dict) -> List[str]:
    if risk_label != 'HIGH':
        return ["No immediate action required. Monitor performance."]
    
    if scenario == "Job Failure":
        actions = ["Reschedule job to off-peak hours."]
        if row.get('MEM_USAGE_PCT', 0) > 80:
            actions.append("Isolate heavy steps and increase memory allocation.")
        if row.get('CONCURRENT_JOBS', 0) > 30:
            actions.append("Reduce job concurrency in the target server group.")
        return actions
    elif scenario == "Transport Failure":
        actions = ["Split transport into smaller logical units."]
        if row.get('OBJ_COUNT', 0) > 200:
            actions.append("Perform a manual peer review of the object list.")
        if row.get('TARGET_SYS_LOAD', 0) > 70:
            actions.append("Validate target system health and wait for lower load.")
        return actions
    elif scenario == "Interface Failure":
        actions = ["Throttle message volume or requeue for later processing."]
        if row.get('PARTNER_RELIABILITY', 1) < 0.8:
            actions.append("Validate partner profile and communication channel.")
        if row.get('QUEUE_DEPTH', 0) > 500:
            actions.append("Investigate destination health and clear queue backlog.")
        return actions
    return ["General investigation required."]

def get_risk_drivers(scenario: str, row: Dict) -> str:
    drivers = []
    if scenario == "Job Failure":
        if row.get('CONCURRENT_JOBS', 0) > 30: drivers.append("High Concurrency")
        if row.get('MEM_USAGE_PCT', 0) > 80: drivers.append("Memory Pressure")
        if row.get('DELAY_SEC', 0) > 200: drivers.append("Start Delay")
    elif scenario == "Transport Failure":
        if row.get('OBJ_COUNT', 0) > 200: drivers.append("Large Object Count")
        if row.get('TABLE_OBJ_PCT', 0) > 0.5: drivers.append("High Table Content")
        if row.get('AUTHOR_SUCCESS_RATE', 1) < 0.85: drivers.append("Low Author Success")
    elif scenario == "Interface Failure":
        if row.get('QUEUE_DEPTH', 0) > 500: drivers.append("Queue Depth")
        if row.get('PARTNER_RELIABILITY', 1) < 0.8: drivers.append("Partner Reliability")
        if row.get('SYS_LOAD_IDX', 0) > 0.7: drivers.append("System Load")
    
    return ", ".join(drivers) if drivers else "Complex Interaction"

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🛡️ SAP Predictive Integrity</h1>
        <p class="header-subtitle">Proactive Operational Risk Prediction for SAP Systems</p>
        <div style="text-align: center; margin-top: 15px;">
            <span class="badge">🔮 Job Failure</span>
            <span class="badge">📦 Transport Risk</span>
            <span class="badge">🔗 Interface Health</span>
        </div>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem; text-align: center; margin-top: 15px;">
            Powered by <strong>SAP-RPT-1</strong> Tabular ML | 1,000 Row Analysis | Actionable Remediation Playbooks
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🛠️ Setup & Data", "🚀 Prediction", "📋 Insights & Export"])

    # ==========================================================================
    # TAB 1: SETUP & DATA
    # ==========================================================================
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### <span class='step-number'>1</span> Model Selection", unsafe_allow_html=True)
            
            model_choice = st.radio(
                "Choose Prediction Model:",
                ["SAP-RPT-1-OSS (Public)", "SAP-RPT-1 (Closed API)"],
                index=0 if st.session_state.model_type == "SAP-RPT-1-OSS (Public)" else 1,
                help="Public model uses HuggingFace. Closed API requires Bearer token."
            )
            
            if model_choice != st.session_state.model_type:
                st.session_state.model_type = model_choice
                st.session_state.token_validated = False
                st.rerun()
            
            st.markdown("---")
            
            if st.session_state.model_type == "SAP-RPT-1-OSS (Public)":
                st.markdown("#### 🤗 HuggingFace Authentication")
                st.markdown("[SAP-RPT-1-OSS on HuggingFace](https://huggingface.co/SAP/sap-rpt-1-oss)")
                hf_token = st.text_input("HuggingFace Token (optional)", 
                                        value=st.session_state.hf_token,
                                        type="password",
                                        help="Optional. Leave blank for public access.")
                
                if st.button("Connect to SAP-RPT-1-OSS", width="stretch"):
                    with st.spinner("Connecting to HuggingFace..."):
                        try:
                            client = SAPRPT1OSSClient(hf_token if hf_token else None)
                            success, msg = client.validate()
                            if success:
                                st.session_state.hf_token = hf_token
                                st.session_state.token_validated = True
                                st.success(msg)
                            else:
                                st.error(msg)
                        except Exception as e:
                            st.error(f"Connection failed: {str(e)}")
            else:
                st.markdown("#### 🔐 SAP-RPT-1 Bearer Token")
                st.markdown("[📄 Get API Token](https://rpt.cloud.sap/docs)", help="Click to open SAP RPT documentation")
                token_input = st.text_input("SAP-RPT-1 Bearer Token", 
                                          value=st.session_state.token, 
                                          type="password",
                                          help="Enter your SAP-RPT-1 API token.")
                st.caption("🔒 **Security Note:** Treat this token like a password. Never commit it to version control or share it in public forums.")
                
                if st.button("Test Connection", width="stretch"):
                    if token_input:
                        client = SAPRPT1Client(token_input)
                        with st.spinner("Validating token..."):
                            success, msg = client.validate_token()
                            if success:
                                st.session_state.token = token_input
                                st.session_state.token_validated = True
                                st.success(f"Validated: ••••{token_input[-4:]}")
                            else:
                                st.error(msg)
                    else:
                        st.warning("Please enter a token.")
            
            if st.session_state.token_validated:
                if st.session_state.model_type == "SAP-RPT-1-OSS (Public)":
                    st.info("✅ Connected to SAP-RPT-1-OSS (HuggingFace)")
                else:
                    st.info(f"Active Token: ••••••••••••{st.session_state.token[-4:]}")
            
            st.markdown("---")
            st.markdown("### <span class='step-number'>2</span> Scenario Selection", unsafe_allow_html=True)
            scenario = st.selectbox("Select Risk Scenario", 
                                  ["Job Failure", "Transport Failure", "Interface Failure"],
                                  index=["Job Failure", "Transport Failure", "Interface Failure"].index(st.session_state.scenario))
            
            if scenario != st.session_state.scenario:
                st.session_state.scenario = scenario
                st.session_state.data = None
                st.session_state.results = None
                st.rerun()

        with col2:
            st.markdown("### <span class='step-number'>3</span> Data Generation", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Scenario:** {st.session_state.scenario}")
                st.write("**Rows:** 1,000")
            with c2:
                if st.button("Rotate Seed & Regenerate", width="stretch"):
                    old_data = st.session_state.data
                    st.session_state.seed = np.random.randint(1, 1000)
                    
                    if st.session_state.scenario == "Job Failure":
                        st.session_state.data = generate_job_failure_data(1000, st.session_state.seed)
                    elif st.session_state.scenario == "Transport Failure":
                        st.session_state.data = generate_transport_failure_data(1000, st.session_state.seed)
                    else:
                        st.session_state.data = generate_interface_failure_data(1000, st.session_state.seed)
                    
                    # Drift detection
                    if old_data is not None:
                        drift_col = 'RISK_SCORE'
                        drift_val = detect_drift(old_data, st.session_state.data, drift_col)
                        if drift_val > 0.15:
                            st.session_state.drift_detected = True
                        else:
                            st.session_state.drift_detected = False
                    
                    st.session_state.results = None
                    st.rerun()

            if st.session_state.drift_detected:
                st.warning("⚠️ **Data Shift Detected!** The new dataset distribution differs significantly from the previous run.")

            if st.session_state.data is not None:
                st.dataframe(st.session_state.data.head(100), width="stretch")
                st.caption(f"Showing first 100 of 1,000 rows. Seed: {st.session_state.seed}")
                
                # ===== SCENARIO DOCUMENTATION =====
                with st.expander("📚 Dataset Schema & SAP Table References", expanded=False):
                    if st.session_state.scenario == "Job Failure":
                        st.markdown("""
### 🔧 Job Failure Prediction Schema

**SAP Tables Referenced:**
| Table | Description | SAP Transaction |
|-------|-------------|-----------------|
| **TBTCO** | Job Header (status, scheduling) | SM37 |
| **TBTCP** | Job Step Parameters | SM37 |
| **TBTCS** | Job Scheduling Details | SM36 |

---

**Column Mapping:**

| Column | Source | Description |
|--------|--------|-------------|
| `JOBNAME` | TBTCO-JOBNAME | Background job name (e.g., Z_MRP_RUN) |
| `JOBCOUNT` | TBTCO-JOBCOUNT | Unique job execution counter |
| `JOBCLASS` | TBTCO-JOBCLASS | Priority class (A=High, B=Medium, C=Low) |
| `DURATION_SEC` | *Derived* | End time - Start time (TBTCO.ENDTIME - TBTCO.STRTTIME) |
| `DELAY_SEC` | *Derived* | Actual start - Scheduled start (queue wait time) |
| `STEP_COUNT` | TBTCP | Count of job steps from TBTCP table |
| `CONCURRENT_JOBS` | *Synthetic* | Simulated count of jobs running at same time |
| `MEM_USAGE_PCT` | *Synthetic* | Simulated memory consumption (real: ST06/SM66) |
| `CPU_LOAD_PCT` | *Synthetic* | Simulated CPU load (real data from ST06) |
| `HAS_VARIANT` | TBTCP-VARIANT | Whether job step uses a variant (1=Yes, 0=No) |
| `HIST_FAIL_RATE` | *Synthetic* | Rolling 30-day failure rate for this JOBNAME |
| `STATUS` | TBTCO-STATUS | Job status: Finished (F) or Cancelled (A) |

---

**⚠️ Synthetic Columns (Not from SAP Tables):**

| Column | Purpose |
|--------|---------|
| `RISK_SCORE` | **Computed risk metric** based on weighted formula combining concurrency, memory, delay, job class, and historical failure rate. Higher = more likely to fail. |
| `RISK_LABEL` | **Derived classification**: HIGH (score > 3.5), MEDIUM (2.2-3.5), LOW (< 2.2). Used as ground truth for model evaluation. |

**Risk Formula:**
```
RISK_SCORE = (CONCURRENT_JOBS/50)*1.5 + (MEM_USAGE_PCT/100)*2.0 
            + (DELAY_SEC/500)*1.2 + (JOBCLASS='A')*0.5 
            + HIST_FAIL_RATE*5.0 + noise
```

> 💡 **Note:** `RISK_SCORE` and `RISK_LABEL` are synthetic labels for demonstration. 
> In production, these would be derived from historical job outcomes or predicted by the model.
                        """)
                        
                    elif st.session_state.scenario == "Transport Failure":
                        st.markdown("""
### 📦 Transport Failure Prediction Schema

**SAP Tables Referenced:**
| Table | Description | SAP Transaction |
|-------|-------------|-----------------|
| **E070** | Transport Header (request info) | SE09/SE10 |
| **E071** | Transport Object List | SE09 |
| **TPLOG** | Transport Logs | STMS |

---

**Column Mapping:**

| Column | Source | Description |
|--------|--------|-------------|
| `TRKORR` | E070-TRKORR | Transport request number (e.g., SIDK900001) |
| `AS4USER` | E070-AS4USER | User who created the transport |
| `OBJ_COUNT` | E071 | Count of objects in the transport |
| `TABLE_OBJ_PCT` | *Derived* | Percentage of table entries (TABU objects) |
| `PROG_OBJ_PCT` | *Derived* | Percentage of programs (PROG/REPS) |
| `CROSS_SYS_DEP` | *Synthetic* | Count of cross-system dependencies |
| `AUTHOR_SUCCESS_RATE` | *Synthetic* | Historical success rate of author's transports |
| `TARGET_SYS_LOAD` | *Synthetic* | Target system CPU/memory load at import |
| `NETWORK_LATENCY` | *Synthetic* | Network latency between source and target |
| `RESULT` | TPLOG | Transport result: Success, Warning, or Error |

---

**⚠️ Synthetic Columns:**

| Column | Purpose |
|--------|---------|
| `RISK_SCORE` | Weighted risk combining object count, table content, author history, and system load. |
| `RISK_LABEL` | HIGH (score > 4.0), MEDIUM (2.5-4.0), LOW (< 2.5) |
                        """)
                        
                    else:  # Interface Failure
                        st.markdown("""
### 🔗 Interface Failure Prediction Schema

**SAP Tables Referenced:**
| Table | Description | SAP Transaction |
|-------|-------------|-----------------|
| **EDIDC** | IDoc Control Record | WE02/WE05 |
| **EDIDS** | IDoc Status Records | WE02 |
| **ARFCSSTATE** | Async RFC Status | SM58 |

---

**Column Mapping:**

| Column | Source | Description |
|--------|--------|-------------|
| `MESTYP` | EDIDC-MESTYP | Message type (ORDERS, INVOIC, MATMAS) |
| `PARTNER` | EDIDC-RCVPRN | Receiving partner logical name |
| `PAYLOAD_SIZE_KB` | EDIDC | IDoc size in kilobytes |
| `QUEUE_DEPTH` | *Synthetic* | Number of IDocs waiting in queue (qRFC) |
| `PARTNER_RELIABILITY` | *Synthetic* | Historical success rate for this partner |
| `RETRY_COUNT` | EDIDS/ARFCSSTATE | Number of retry attempts |
| `SYS_LOAD_IDX` | *Synthetic* | System load index (0-1 scale) |
| `DEST_AVAILABILITY` | *Synthetic* | RFC destination availability (0-1 scale) |
| `STATUS_CODE` | EDIDS-STATUS | IDoc status (53=Success, 51/61=Error) |

---

**⚠️ Synthetic Columns:**

| Column | Purpose |
|--------|---------|
| `RISK_SCORE` | Weighted risk combining queue depth, partner reliability, payload size, and system load. |
| `RISK_LABEL` | HIGH (score > 3.8), MEDIUM (2.0-3.8), LOW (< 2.0) |
                        """)
                    
                    st.markdown("""
---
### 🎯 Understanding RISK_SCORE and RISK_LABEL

These columns are **not from SAP tables** — they are synthetic labels computed using a non-linear formula 
that combines multiple risk factors. They serve two purposes:

1. **Ground Truth for Evaluation**: Compare the model's predictions (`PRED_LABEL`) against these synthetic labels to measure accuracy.
   
2. **Training Signal**: If using SAP-RPT-1-OSS, these labels can serve as the target variable for the classifier.

> ⚠️ **Important**: In production scenarios, replace these synthetic labels with actual historical outcomes 
> (e.g., did the job actually fail?) from your SAP system.
                    """)
            else:
                st.info("Click 'Rotate Seed & Regenerate' to build the synthetic dataset.")

    # ==========================================================================
    # TAB 2: PREDICTION
    # ==========================================================================
    with tab2:
        if not st.session_state.token_validated:
            st.warning("⚠️ Please connect to a model in Tab 1 first.")
            if st.button("Run in Offline Mode (Mock Predictions)"):
                st.session_state.token_validated = True
                st.session_state.token = "MOCK_TOKEN"
                st.session_state.model_type = "Offline"
                st.rerun()
        elif st.session_state.data is None:
            st.warning("⚠️ Please generate data in Tab 1 first.")
        else:
            st.markdown("### <span class='step-number'>4</span> Execute Scoring", unsafe_allow_html=True)
            st.info(f"**Model:** {st.session_state.model_type}")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🚀 Score 1,000 Rows", type="primary", width="stretch"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(p):
                        progress_bar.progress(p)
                        status_text.text(f"Scoring Progress: {int(p*100)}%")

                    try:
                        with st.spinner("Running prediction..."):
                            if st.session_state.token == "MOCK_TOKEN" or st.session_state.model_type == "Offline":
                                # Mock mode
                                client = SAPRPT1Client("MOCK")
                                predictions = client.mock_predict(st.session_state.data)
                            
                            elif st.session_state.model_type == "SAP-RPT-1-OSS (Public)":
                                # Use HuggingFace TabPFN
                                client = SAPRPT1OSSClient(st.session_state.hf_token if st.session_state.hf_token else None)
                                
                                # Split data: use first 200 rows as training, rest as test
                                train_size = min(200, len(st.session_state.data) // 5)
                                train_df = st.session_state.data.head(train_size)
                                test_df = st.session_state.data.tail(len(st.session_state.data) - train_size)
                                
                                # Get feature columns (exclude label columns)
                                exclude_cols = ['STATUS', 'RISK_SCORE', 'RISK_LABEL', 'RESULT', 'JOBCOUNT', 'TRKORR', 'JOBNAME', 'AS4USER', 'MESTYP', 'PARTNER']
                                feature_cols = [c for c in st.session_state.data.columns if c not in exclude_cols]
                                
                                predictions = client.predict_from_df(
                                    train_df, test_df, feature_cols, 'RISK_LABEL',
                                    progress_callback=update_progress
                                )
                                
                                # Pad predictions for the training rows (use ground truth)
                                train_preds = [{"label": row['RISK_LABEL'], "probability": 0.99, "score": row['RISK_SCORE']} 
                                              for _, row in train_df.iterrows()]
                                predictions = train_preds + predictions
                            
                            else:
                                # Use closed SAP-RPT-1 API
                                client = SAPRPT1Client(st.session_state.token)
                                features_df = st.session_state.data.drop(columns=['STATUS', 'RISK_SCORE', 'RISK_LABEL', 'RESULT'], errors='ignore')
                                predictions = client.predict_full(features_df, batch_size=100, progress_callback=update_progress)
                            
                            # Merge results
                            results_df = st.session_state.data.copy()
                            pred_labels = [p['label'] for p in predictions]
                            pred_probs = [p['probability'] for p in predictions]
                            
                            results_df['PRED_LABEL'] = pred_labels
                            results_df['CONFIDENCE'] = pred_probs
                            
                            st.session_state.results = results_df
                            st.success("Scoring complete!")
                    except Exception as e:
                        st.error(f"Scoring failed: {str(e)}")

            with col2:
                if st.session_state.results is not None:
                    high_risk_count = len(st.session_state.results[st.session_state.results['PRED_LABEL'] == 'HIGH'])
                    st.metric("High Risk Entities Detected", f"{high_risk_count} / 1,000", delta=f"{high_risk_count/10}%", delta_color="inverse")

            if st.session_state.results is not None:
                st.markdown("---")
                st.markdown("#### Top 100 High-Risk Predictions")
                
                # Sort by confidence for high risk
                top_100 = st.session_state.results.sort_values(by=['PRED_LABEL', 'CONFIDENCE'], ascending=[True, False]).head(100)
                
                def color_risk(val):
                    if val == 'HIGH': return 'background-color: rgba(255, 75, 75, 0.2)'
                    if val == 'MEDIUM': return 'background-color: rgba(255, 165, 0, 0.2)'
                    return ''

                st.dataframe(top_100.style.map(color_risk, subset=['PRED_LABEL']), width="stretch")
                
                with st.expander("View Full 1,000 Row Results (Scrolled Pagination)"):
                    st.dataframe(st.session_state.results, width="stretch")

    # ==========================================================================
    # TAB 3: INSIGHTS & EXPORT
    # ==========================================================================
    with tab3:
        if st.session_state.results is None:
            st.error("❌ No results found. Please run scoring in Tab 2.")
        else:
            st.markdown("### <span class='step-number'>5</span> Remediation Playbooks", unsafe_allow_html=True)
            
            high_risk_df = st.session_state.results[st.session_state.results['PRED_LABEL'] == 'HIGH'].head(5)
            
            if high_risk_df.empty:
                st.success("✅ No HIGH risk entities detected in this run.")
            else:
                for _, row in high_risk_df.iterrows():
                    entity_id = row.get('JOBNAME') or row.get('TRKORR') or row.get('MESTYP')
                    with st.expander(f"🚨 High Risk: {entity_id} (Confidence: {row['CONFIDENCE']:.1%})"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Risk Drivers:**")
                            st.write(get_risk_drivers(st.session_state.scenario, row))
                        with c2:
                            st.markdown("**Suggested Actions:**")
                            for action in get_remediation_playbook(st.session_state.scenario, 'HIGH', row):
                                st.write(f"- {action}")

            st.markdown("---")
            st.markdown("### <span class='step-number'>6</span> Export & Audit", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                csv = st.session_state.results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Scored Dataset (CSV)",
                    csv,
                    f"sap_risk_results_{st.session_state.scenario.lower().replace(' ', '_')}.csv",
                    "text/csv",
                    width="stretch"
                )
            
            with c2:
                audit_log = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "scenario": st.session_state.scenario,
                    "seed": st.session_state.seed,
                    "row_count": 1000,
                    "high_risk_count": int(len(st.session_state.results[st.session_state.results['PRED_LABEL'] == 'HIGH'])),
                    "token_masked": f"••••{st.session_state.token[-4:]}" if st.session_state.token else "NONE"
                }
                st.download_button(
                    "Download Run Audit (JSON)",
                    json.dumps(audit_log, indent=2),
                    "run_audit.json",
                    "application/json",
                    width="stretch"
                )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-top: 30px;">
        <p style="color: white; font-size: 15px; margin: 0;">
            🛡️ <strong>SAP Predictive Integrity</strong> | Developed by <strong>Amit Lal</strong> | 
            <a href="https://aka.ms/amitlal" target="_blank" style="color: #fff; text-decoration: underline;">aka.ms/amitlal</a>
        </p>
        <p style="color: rgba(255,255,255,0.85); font-size: 12px; margin: 8px 0 0 0;">
            Proactive Risk Detection for SAP Background Jobs, Transports & Interfaces using SAP-RPT-1 Tabular ML
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <p style="text-align: center; font-size: 11px; color: #6c757d; margin-top: 15px; padding: 0 20px; line-height: 1.6;">
        ⚖️ <strong>Disclaimer:</strong> SAP, SAP RPT, SAP-RPT-1, and all SAP logos and product names are trademarks or registered trademarks of SAP SE in Germany and other countries. 
        This is an independent demonstration project for educational purposes only and is not affiliated with, endorsed by, or sponsored by SAP SE or any enterprise. 
        The synthetic datasets used in this application are for demonstration purposes only and do not represent real SAP system data. 
        All other trademarks are the property of their respective owners.
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

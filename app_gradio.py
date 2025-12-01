"""
SAP Finance Dashboard with RPT-1-OSS Model - Gradio Version

Main Gradio application with tabs:
- Dashboard: Overview with metrics and charts
- Data Explorer: Browse datasets
- Upload: Upload custom datasets
- Predictions: AI-powered predictions using SAP-RPT-1-OSS
- OData: Connect to SAP OData services
"""

import gradio as gr
print(f"Gradio version: {gr.__version__}")
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import utilities
from utils.data_generator import generate_all_datasets
from utils.visualizations import (
    create_revenue_expense_chart,
    create_balance_sheet_chart,
    create_gl_summary_chart,
    create_sales_analytics_chart,
    create_sales_trend_chart,
    get_summary_metrics,
    create_prediction_distribution_chart,
    create_prediction_bar_chart,
    create_confidence_gauge
)
from utils.odata_connector import SAPFinanceConnector
from models.rpt_model import create_model
from utils.playground import (
    load_dataset,
    detect_task_type,
    detect_task_type_from_column,
    get_dataset_info,
    auto_select_target_column,
    prepare_train_test_split,
    preprocess_data,
    export_results,
    check_embedding_server,
    start_embedding_server,
    ensure_embedding_server_running,
    is_sap_rpt_oss_installed
)

# Load environment variables
load_dotenv()

# Global variables
gl_data = pd.DataFrame()
financial_data = pd.DataFrame()
sales_data = pd.DataFrame()
uploaded_data = pd.DataFrame()
odata_data = pd.DataFrame()
odata_connector = None
model_wrapper = None

# Playground variables
playground_data = pd.DataFrame()
playground_model = None
playground_results = None


def load_datasets():
    """Load synthetic datasets if they exist."""
    global gl_data, financial_data, sales_data
    data_dir = Path("data")
    
    if not data_dir.exists():
        generate_all_datasets()
    
    if (data_dir / "synthetic_gl_accounts.csv").exists():
        gl_data = pd.read_csv(data_dir / "synthetic_gl_accounts.csv")
    
    if (data_dir / "synthetic_financial_statements.csv").exists():
        financial_data = pd.read_csv(data_dir / "synthetic_financial_statements.csv")
    
    if (data_dir / "synthetic_sales_orders.csv").exists():
        sales_data = pd.read_csv(data_dir / "synthetic_sales_orders.csv")


def create_dashboard():
    """Create dashboard with metrics and charts."""
    if gl_data.empty and financial_data.empty and sales_data.empty:
        load_datasets()
    
    # Calculate metrics with vibrant styling
    metrics_html = "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 30px;'>"
    
    if not gl_data.empty:
        gl_metrics = get_summary_metrics(gl_data, "gl")
        metrics_html += f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white;'>
            <h3 style='margin: 0 0 10px 0; font-size: 16px; opacity: 0.9;'>💰 GL Transactions</h3>
            <p style='font-size: 32px; font-weight: bold; margin: 0;'>{gl_metrics.get('Total Transactions', 0):,}</p>
        </div>
        """
    
    if not financial_data.empty:
        fin_metrics = get_summary_metrics(financial_data, "financial")
        metrics_html += f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white;'>
            <h3 style='margin: 0 0 10px 0; font-size: 16px; opacity: 0.9;'>📈 Latest Revenue</h3>
            <p style='font-size: 32px; font-weight: bold; margin: 0;'>${fin_metrics.get('Latest Revenue', 0):,.0f}</p>
        </div>
        """
    
    if not sales_data.empty:
        sales_metrics = get_summary_metrics(sales_data, "sales")
        metrics_html += f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white;'>
            <h3 style='margin: 0 0 10px 0; font-size: 16px; opacity: 0.9;'>🛒 Total Sales</h3>
            <p style='font-size: 32px; font-weight: bold; margin: 0;'>${sales_metrics.get('Total Sales', 0):,.0f}</p>
        </div>
        """
    
    datasets_count = sum([not df.empty for df in [gl_data, financial_data, sales_data, uploaded_data]])
    metrics_html += f"""
    <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); color: white;'>
        <h3 style='margin: 0 0 10px 0; font-size: 16px; opacity: 0.9;'>📊 Datasets</h3>
        <p style='font-size: 32px; font-weight: bold; margin: 0;'>{datasets_count} loaded</p>
    </div>
    </div>
    """
    
    # Create charts
    charts = []
    if not financial_data.empty:
        fig_dict = create_revenue_expense_chart(financial_data)
        if fig_dict:
            charts.append(go.Figure(fig_dict))
        
        fig_dict = create_balance_sheet_chart(financial_data)
        if fig_dict:
            charts.append(go.Figure(fig_dict))
    
    if not sales_data.empty:
        fig_dict = create_sales_analytics_chart(sales_data)
        if fig_dict:
            charts.append(go.Figure(fig_dict))
    
    return metrics_html, charts[0] if len(charts) > 0 else None, charts[1] if len(charts) > 1 else None, charts[2] if len(charts) > 2 else None


def explore_dataset(dataset_type):
    """Explore selected dataset."""
    global gl_data, financial_data, sales_data, uploaded_data
    
    if dataset_type == "GL Accounts":
        if gl_data.empty:
            return "No GL data available", None, None
        fig_dict = create_gl_summary_chart(gl_data)
        fig = go.Figure(fig_dict) if fig_dict else None
        return f"GL Accounts ({len(gl_data)} records)", fig, gl_data.head(100)
    
    elif dataset_type == "Financial Statements":
        if financial_data.empty:
            return "No financial data available", None, None
        fig_dict = create_revenue_expense_chart(financial_data)
        fig = go.Figure(fig_dict) if fig_dict else None
        return f"Financial Statements ({len(financial_data)} records)", fig, financial_data
    
    elif dataset_type == "Sales Orders":
        if sales_data.empty:
            return "No sales data available", None, None
        fig_dict = create_sales_trend_chart(sales_data)
        fig = go.Figure(fig_dict) if fig_dict else None
        return f"Sales Orders ({len(sales_data)} records)", fig, sales_data.head(100)
    
    elif dataset_type == "Uploaded Data":
        if uploaded_data.empty:
            return "No uploaded data available", None, None
        return f"Uploaded Data ({len(uploaded_data)} records)", None, uploaded_data.head(100)
    
    return "Select a dataset", None, None


def upload_file(file):
    """Handle file upload."""
    global uploaded_data
    if file is not None:
        try:
            uploaded_data = pd.read_csv(file.name)
            return f"Successfully uploaded {len(uploaded_data)} records!", uploaded_data.head(50)
        except Exception as e:
            return f"Error uploading file: {str(e)}", None
    return "No file uploaded", None


def init_model(model_type, use_gpu):
    """Initialize the SAP-RPT-1-OSS model."""
    global model_wrapper
    try:
        model_wrapper = create_model(model_type=model_type.lower(), use_gpu=use_gpu)
        
        context_size = 8192 if use_gpu else 2048
        bagging = 8 if use_gpu else 1
        
        return f"""✅ SAP-RPT-1-OSS Model Initialized Successfully!

🎯 Model Type: {model_type}
🔧 Context Size: {context_size}
📦 Bagging Factor: {bagging}
💻 Mode: {'GPU (80GB)' if use_gpu else 'CPU (Lightweight)'}
📝 Status: Ready for training

⚠️ Requirements:
   • Hugging Face authentication
   • Embedding service (may be required for predictions)
   • Sufficient memory"""
    except ImportError as e:
        return f"""❌ SAP-RPT-1-OSS Model Not Available

Error: {str(e)}

📋 Installation Required:
   pip install git+https://github.com/SAP-samples/sap-rpt-1-oss

🔑 Authentication Required:
   1. Create Hugging Face account
   2. Accept model license at: https://huggingface.co/SAP/sap-rpt-1-oss
   3. Run: huggingface-cli login
   4. Set HUGGINGFACE_TOKEN in .env file"""
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        
        # Check for common errors
        if "HUGGINGFACE_TOKEN" in str(e) or "login" in str(e).lower():
            return f"""❌ Hugging Face Authentication Failed

Error: {str(e)}

🔑 Required Steps:
   1. Login to Hugging Face: huggingface-cli login
   2. OR set HUGGINGFACE_TOKEN in .env file
   3. Accept model terms: https://huggingface.co/SAP/sap-rpt-1-oss"""
        
        elif "memory" in str(e).lower() or "cuda" in str(e).lower():
            return f"""❌ Insufficient Resources

Error: {str(e)}

💻 Requirements:
   • GPU with 80GB memory (recommended)
   • OR use CPU mode (uncheck GPU option)
   • Context size will be reduced for CPU mode"""
        
        else:
            return f"""❌ SAP-RPT-1-OSS Initialization Failed

Error: {str(e)}

📋 Details:
{error_detail[:500]}

🔧 Common Solutions:
   1. Ensure model is installed
   2. Check Hugging Face authentication
   3. Verify system resources
   4. Try CPU mode if GPU unavailable"""


def train_model(dataset_type):
    """Train the model on selected dataset."""
    global model_wrapper, gl_data, financial_data, sales_data, uploaded_data
    
    if model_wrapper is None:
        return "Please initialize the model first"
    
    # Select dataset
    if dataset_type == "GL Accounts":
        df = gl_data
    elif dataset_type == "Financial Statements":
        df = financial_data
    elif dataset_type == "Sales Orders":
        df = sales_data
    elif dataset_type == "Uploaded Data":
        df = uploaded_data
    else:
        return "Please select a dataset"
    
    if df.empty:
        return "Selected dataset is empty"
    
    try:
        # Get numeric columns and clean data
        X = df.select_dtypes(include=[np.number])
        
        # Remove columns with all NaN values
        X = X.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with 0
        X = X.fillna(0)
        
        if len(X) > 0 and len(X.columns) > 0:
            # Create a simple target for classification based on first column
            y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
            
            # Keep as DataFrame - SAP-RPT-OSS expects DataFrame or compatible format
            X_train = pd.DataFrame(X, columns=X.columns)
            X_train = X_train.astype(float)
            
            # Fit the model with DataFrame
            model_wrapper.fit(X_train, y)
            return f"✅ Model trained successfully on {len(X)} samples with {len(X.columns)} features!"
        else:
            return "No numeric data available for training"
    except Exception as e:
        return f"Error training model: {str(e)}"


def get_scenario_labels(dataset_type, scenario):
    """Get contextual labels for predictions based on dataset and scenario."""
    labels_map = {
        "Sales Orders": {
            "High Value Order Classification": {
                0: "Standard Order (Low Value)",
                1: "High Value Order (Premium)",
                "description": "Identifies orders with high revenue potential"
            },
            "Order Priority Classification": {
                0: "Normal Priority",
                1: "High Priority / Urgent",
                "description": "Flags orders requiring immediate attention"
            },
            "Customer Segment Classification": {
                0: "Regular Customer",
                1: "VIP / Enterprise Customer",
                "description": "Identifies high-value customer segments"
            }
        },
        "Products": {
            "Product Performance Classification": {
                0: "Low Performer",
                1: "Top Performer / Best Seller",
                "description": "Identifies products with high sales performance"
            },
            "Stock Risk Classification": {
                0: "Normal Stock Level",
                1: "Low Stock / Reorder Needed",
                "description": "Flags products at risk of stockout"
            }
        },
        "GL Accounts": {
            "Transaction Risk Classification": {
                0: "Normal Transaction",
                1: "Flagged / Review Needed",
                "description": "Identifies potentially risky or unusual transactions"
            },
            "Account Balance Classification": {
                0: "Below Average Balance",
                1: "Above Average Balance",
                "description": "Classifies accounts by balance magnitude"
            },
            "Expense Category Classification": {
                0: "Operating Expense",
                1: "Capital Expenditure",
                "description": "Categorizes transactions by type"
            }
        },
        "Financial Statements": {
            "Financial Health Classification": {
                0: "Below Average Performance",
                1: "Strong Performance",
                "description": "Assesses overall financial health"
            },
            "Profitability Classification": {
                0: "Low Margin Period",
                1: "High Margin Period",
                "description": "Identifies periods with strong profitability"
            },
            "Growth Trend Classification": {
                0: "Declining Revenue",
                1: "Revenue Growth",
                "description": "Classifies periods by revenue trajectory"
            }
        }
    }
    
    default_labels = {
        0: "Class 0 (Negative/Low)",
        1: "Class 1 (Positive/High)",
        "description": "Binary classification"
    }
    
    return labels_map.get(dataset_type, {}).get(scenario, default_labels)


def make_predictions(dataset_type, prediction_scenario):
    """Make predictions on selected dataset with scenario context."""
    global model_wrapper, gl_data, financial_data, sales_data, uploaded_data
    
    if model_wrapper is None:
        return "❌ Please initialize the model first", None
    
    if not hasattr(model_wrapper, 'is_fitted') or not model_wrapper.is_fitted:
        return "❌ Please train the model first", None
    
    # Select dataset and get original data for context
    if dataset_type == "Sales Orders":
        df = sales_data.copy()
        original_cols = ['Order_Number', 'Customer_Name', 'Total_Amount', 'Status']
    elif dataset_type == "Products":
        df = sales_data.copy()
        original_cols = ['Product_Name', 'Total_Amount', 'Quantity']
    elif dataset_type == "GL Accounts":
        df = gl_data.copy()
        original_cols = ['Transaction_ID', 'Account_Description', 'Debit', 'Credit']
    elif dataset_type == "Financial Statements":
        df = financial_data.copy()
        original_cols = ['Period', 'Revenue', 'Net_Income']
    elif dataset_type == "Uploaded Data":
        df = uploaded_data.copy()
        original_cols = df.columns[:3].tolist() if len(df.columns) >= 3 else df.columns.tolist()
    else:
        return "Please select a dataset", None
    
    if df.empty:
        return f"❌ Selected dataset ({dataset_type}) is empty", None
    
    try:
        # Get labels for this scenario
        label_config = get_scenario_labels(dataset_type, prediction_scenario)
        
        # Get numeric columns
        X = df.select_dtypes(include=[np.number])
        X = X.dropna(axis=1, how='all')
        X = X.fillna(X.mean())
        
        if len(X) > 0 and len(X.columns) > 0:
            # Limit to first 15 rows
            X_sample = X.head(15)
            
            # Keep as DataFrame with proper column names - SAP-RPT-OSS expects DataFrame
            X_pred = pd.DataFrame(X_sample, columns=X.columns)
            
            # Ensure all values are numeric and no NaN
            X_pred = X_pred.astype(float)
            X_pred = X_pred.fillna(0)
            
            # Make predictions - pass DataFrame directly
            predictions = model_wrapper.predict(X_pred)
            
            # Convert to numpy array and flatten if needed
            predictions = np.array(predictions)
            if hasattr(predictions, 'flatten') and len(predictions.shape) > 1:
                predictions = predictions.flatten()
            
            # Get original data columns for context
            context_df = df.head(15)[original_cols] if all(col in df.columns for col in original_cols) else df.head(15).iloc[:, :3]
            
            # Create result with meaningful labels
            model_type = model_wrapper.model_type.capitalize()
            
            if model_type == "Classifier":
                pred_labels = [label_config.get(int(p), f"Class {int(p)}") for p in predictions]
                
                result_df = pd.DataFrame({
                    'Row': range(1, len(predictions) + 1),
                    'Prediction': pred_labels,
                    'Confidence': predictions
                })
                
                # Add context columns
                for col in context_df.columns:
                    result_df[col] = context_df[col].values
                
                # Count predictions
                class_0_count = sum(predictions == 0)
                class_1_count = sum(predictions == 1)
                
                # Create visualizations
                pie_chart = go.Figure(create_prediction_distribution_chart(
                    predictions, 
                    label_config, 
                    f"{prediction_scenario} - Distribution"
                ))
                
                bar_chart = go.Figure(create_prediction_bar_chart(
                    predictions,
                    label_config,
                    f"{prediction_scenario} - Summary"
                ))
                
                # Calculate confidence score
                confidence = max(class_0_count, class_1_count) / len(predictions) * 100
                gauge_chart = go.Figure(create_confidence_gauge(
                    confidence,
                    "Prediction Confidence"
                ))
                
                status = f"""✅ {model_type} Results - {prediction_scenario}

📊 {label_config.get('description', 'Classification complete')}

Analyzed {len(predictions)} records:
  • {label_config.get(1, 'Class 1')}: {class_1_count} records ({class_1_count/len(predictions)*100:.1f}%)
  • {label_config.get(0, 'Class 0')}: {class_0_count} records ({class_0_count/len(predictions)*100:.1f}%)

Dataset: {dataset_type}
Model Type: {model_type}
Confidence: {confidence:.1f}%"""
            else:
                result_df = pd.DataFrame({
                    'Row': range(1, len(predictions) + 1),
                    'Predicted Value': predictions.round(2)
                })
                
                # Add context columns
                for col in context_df.columns:
                    result_df[col] = context_df[col].values
                
                # Create visualizations for regression
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(predictions) + 1)),
                    y=predictions,
                    mode='lines+markers',
                    marker=dict(size=10, color='#3498db'),
                    line=dict(width=3, color='#3498db')
                ))
                fig.update_layout(
                    title=f"{prediction_scenario} - Predicted Values",
                    xaxis_title="Sample",
                    yaxis_title="Predicted Value",
                    template='plotly_white',
                    height=400
                )
                pie_chart = fig
                bar_chart = None
                gauge_chart = None
                
                status = f"""✅ {model_type} Results - {prediction_scenario}

Predicted {len(predictions)} values
Mean: {predictions.mean():.2f}
Range: {predictions.min():.2f} to {predictions.max():.2f}
Std Dev: {predictions.std():.2f}

Dataset: {dataset_type}"""
            
            return status, result_df, pie_chart, bar_chart, gauge_chart
        else:
            return f"❌ No valid numeric data available in {dataset_type}", None, None, None, None
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        
        # Check for specific SAP-RPT-1-OSS errors
        if "zmq" in str(e).lower() or "socket" in str(e).lower() or "Resource temporarily unavailable" in str(e):
            return f"""❌ SAP-RPT-1-OSS Embedding Service Not Available

Error: {str(e)}

🔧 SAP-RPT-1-OSS requires an embedding service to be running:

**Required Setup:**
1. The model uses a text embedding service via ZMQ socket
2. This service needs to be started separately
3. Service handles semantic understanding of column names and values

**To Use SAP-RPT-1-OSS:**
• Start the embedding service (see SAP-RPT-1-OSS documentation)
• Ensure ZMQ socket is accessible
• Verify service is running before making predictions

**Current Status:** Model initialized but embedding service unavailable

📖 Documentation: https://github.com/SAP-samples/sap-rpt-1-oss
🔗 Model Info: https://huggingface.co/SAP/sap-rpt-1-oss

Dataset: {dataset_type}
Scenario: {prediction_scenario}""", None, None, None, None
        
        else:
            return f"""❌ Error making predictions on {dataset_type}

Error: {str(e)}

📋 Details:
{error_detail[:400]}

Dataset: {dataset_type}
Scenario: {prediction_scenario}""", None, None, None, None


def update_scenarios(dataset_type):
    """Update scenario dropdown based on selected dataset."""
    scenarios_map = {
        "Sales Orders": [
            "High Value Order Classification",
            "Order Priority Classification",
            "Customer Segment Classification"
        ],
        "Products": [
            "Product Performance Classification",
            "Stock Risk Classification"
        ],
        "GL Accounts": [
            "Transaction Risk Classification",
            "Account Balance Classification",
            "Expense Category Classification"
        ],
        "Financial Statements": [
            "Financial Health Classification",
            "Profitability Classification",
            "Growth Trend Classification"
        ],
        "Uploaded Data": [
            "Custom Classification"
        ]
    }
    
    scenarios = scenarios_map.get(dataset_type, ["Custom Classification"])
    return gr.Dropdown(choices=scenarios, value=scenarios[0])


def test_odata_connection():
    """Test OData connection."""
    global odata_connector
    try:
        odata_connector = SAPFinanceConnector()
        connected, message = odata_connector.test_connection()
        if connected:
            return f"✓ {message}"
        else:
            return f"✗ {message}"
    except Exception as e:
        return f"Error: {str(e)}"


def fetch_odata_data(entity_type, num_records):
    """Fetch data from OData service."""
    global odata_connector, odata_data
    
    if odata_connector is None:
        return "Please test connection first", None
    
    try:
        if entity_type == "Sales Orders":
            odata_data = odata_connector.fetch_orders_df(num_records)
        elif entity_type == "Products":
            odata_data = odata_connector.fetch_products_df(num_records)
        elif entity_type == "Line Items":
            odata_data = odata_connector.fetch_line_items_df(num_records)
        elif entity_type == "Business Partners":
            odata_data = odata_connector.fetch_partners_df(num_records)
        else:
            return "Please select an entity type", None
        
        return f"Fetched {len(odata_data)} records", odata_data.head(100) if not odata_data.empty else None
    except Exception as e:
        return f"Error fetching data: {str(e)}", None


# Playground functions
def handle_playground_upload(file):
    """Handle dataset upload in playground."""
    global playground_data
    
    if file is None:
        return "No file uploaded", None, [], None, "classification", [], None
    
    try:
        df, error = load_dataset(file.name)
        if error:
            return f"Error: {error}", None, [], None, "classification", [], None
        
        playground_data = df
        
        # Get dataset info
        info = get_dataset_info(df)
        
        # Auto-select target column (default to last)
        target_col = auto_select_target_column(df, "classification")
        
        # Detect task type from filename first
        filename_task_type = detect_task_type(Path(file.name).name)
        
        # Then detect from target column data type
        column_task_type = detect_task_type_from_column(df, target_col)
        
        # Use column-based detection if filename detection is default
        if filename_task_type == "classification" and column_task_type == "regression":
            task_type = column_task_type  # Prefer column-based detection
        else:
            task_type = filename_task_type
        
        # Create info text
        target_info = ""
        if target_col:
            target_series = df[target_col]
            if pd.api.types.is_numeric_dtype(target_series):
                unique_count = target_series.dropna().nunique()
                target_info = f"\nTarget '{target_col}': {unique_count} unique values"
                if unique_count > 20:
                    target_info += " (suggests regression)"
                else:
                    target_info += " (suggests classification)"
        
        info_text = f"""Dataset loaded successfully!

Rows: {info['num_rows']:,}
Columns: {info['num_columns']}
Numeric columns: {len(info['numeric_columns'])}
Categorical columns: {len(info['categorical_columns'])}

Detected task type: {task_type} (from filename: {filename_task_type}, from column: {column_task_type})
Suggested target column: {target_col}{target_info}"""
        
        # Preview first 10 rows
        preview = df.head(10)
        
        # Column list for dropdown
        columns = list(df.columns)
        
        return (
            info_text,
            preview,
            columns,  # Choices for dropdown
            target_col,  # Value for dropdown
            task_type,
            columns,  # Choices for second dropdown
            target_col  # Value for second dropdown
        )
    except Exception as e:
        return f"Error: {str(e)}", None, [], None, "classification", [], None


def train_playground_model(
    task_type,
    target_column,
    test_split,
    max_context_size,
    bagging,
    use_gpu,
    handle_missing,
    normalize,
    progress=gr.Progress()
):
    """Train model in playground with progress tracking."""
    global playground_data, playground_model
    
    if playground_data.empty:
        return "Please upload a dataset first", None, None, None
    
    try:
        progress(0.1, desc="Preparing data...")
        
        # Preprocess data
        df_processed = preprocess_data(playground_data, handle_missing, normalize)
        
        progress(0.2, desc="Validating target column...")
        
        # Validate target column exists
        if target_column not in df_processed.columns:
            return f"Error: Target column '{target_column}' not found in dataset", None, None, None
        
        # Check target column data type
        target_series = df_processed[target_column]
        target_dtype = target_series.dtype
        
        # Auto-detect task type if mismatch
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        is_integer_like = False
        
        if is_numeric:
            # Check if it's integer-like (can be converted to int without loss)
            try:
                int_values = target_series.dropna().astype(int)
                float_values = target_series.dropna().astype(float)
                is_integer_like = (int_values == float_values).all()
            except:
                is_integer_like = False
        
        # Validate task type matches target column
        if task_type == "classification":
            if not is_integer_like:
                # Check if it's numeric with many unique values
                if is_numeric:
                    unique_values = target_series.dropna().nunique()
                    if unique_values > 20:  # Too many unique values for classification
                        return f"""Error: Target column '{target_column}' contains continuous numeric values ({unique_values} unique values).

This looks like a regression problem, not classification.

Solution: Change Task Type to 'regression' or convert your target to integer classes.""", None, None, None
                    else:
                        # Convert numeric to integer classes (will be handled later with LabelEncoder)
                        pass
                else:
                    # String/categorical - will be encoded with LabelEncoder later
                    # No need to convert here, just validate
                    unique_values = target_series.dropna().nunique()
                    if unique_values > 100:
                        return f"""Error: Target column '{target_column}' has too many unique categories ({unique_values}).

Classification works best with fewer categories (< 100).

Solution: Consider grouping categories or using regression if this is a continuous value.""", None, None, None
        else:  # regression
            if not is_numeric:
                return f"""Error: Target column '{target_column}' is not numeric (type: {target_dtype}).

Regression requires numeric target values.

Solution: Change Task Type to 'classification' or convert your target to numeric.""", None, None, None
        
        progress(0.3, desc="Splitting train/test...")
        
        # Prepare train/test split
        X_train, y_train, X_test, y_test = prepare_train_test_split(
            df_processed, target_column, test_split
        )
        
        # Ensure classification targets are integers
        if task_type == "classification":
            # Handle string/categorical targets by encoding them
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = pd.Series(le.fit_transform(y_train.astype(str)), index=y_train.index)
            y_test = pd.Series(le.transform(y_test.astype(str)), index=y_test.index)
        
        progress(0.4, desc="Preparing model...")
        
        # Note: SAP-RPT-OSS typically starts the embedding server automatically when needed
        # We check status but don't require it to be running beforehand
        server_running, server_msg = ensure_embedding_server_running()
        server_warning = ""
        if not server_running:
            # This is normal - the model will start the server automatically when making predictions
            server_warning = f"\n💡 Note: Embedding server will start automatically when model makes predictions."
        
        progress(0.5, desc="Initializing model...")
        
        # Initialize model with custom parameters
        model_type = "classifier" if task_type == "classification" else "regressor"
        from models.rpt_model import RPTModelWrapper
        playground_model = RPTModelWrapper(
            model_type=model_type,
            max_context_size=max_context_size,
            bagging=bagging
        )
        
        progress(0.6, desc="Training model...")
        
        # Train model
        playground_model.fit(X_train, y_train)
        
        progress(0.8, desc="Making predictions...")
        
        # Make predictions
        predictions = playground_model.predict(X_test)
        
        progress(0.9, desc="Exporting results...")
        
        # Export results
        results_path = export_results(
            X_test, y_test, predictions, task_type,
            filename_prefix="playground"
        )
        
        progress(1.0, desc="Complete!")
        
        # Calculate metrics
        if task_type == "classification":
            accuracy = (predictions == y_test.values).mean() * 100
            metrics = f"Accuracy: {accuracy:.2f}%"
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            metrics = f"MSE: {mse:.4f}, R²: {r2:.4f}"
        
        # Create results DataFrame for display
        results_df = X_test.copy()
        results_df['true_value'] = y_test.values
        if task_type == "classification":
            results_df['predicted_class'] = predictions
        else:
            results_df['predicted_value'] = predictions
        
        status = f"""✅ Training Complete!

Training samples: {len(X_train):,}
Test samples: {len(X_test):,}
{metrics}
{server_warning}

Results exported to: {results_path}"""
        
        return status, results_df.head(100), results_path, gr.File(value=results_path)
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"Error: {str(e)}\n\nDetails:\n{error_detail[:500]}", None, None, None


def check_playground_embedding_server():
    """Check embedding server status."""
    # First check if package is installed
    if not is_sap_rpt_oss_installed():
        return f"❌ sap-rpt-oss package not found\n\n📦 Installation Required:\n1. Install sap-rpt-oss: pip install git+https://github.com/SAP-samples/sap-rpt-1-oss\n2. Install pyzmq: pip install pyzmq\n\n💡 After installation, the server will auto-start when you train a model."
    
    # Check if server is running
    is_running, message = check_embedding_server()
    if is_running:
        return f"✅ {message}\n\nThe embedding server is ready to use."
    else:
        return f"ℹ️ {message}\n\n✅ This is normal! The embedding server will start automatically when you train a model or make predictions. No manual start needed."


# Create Gradio interface with vibrant theme
with gr.Blocks(title="SAP Finance Dashboard") as app:
    gr.HTML("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1 style='font-size: 42px; margin-bottom: 10px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            📊SAP Finance playground for RPT-1-OSS Model
        </h1>
        <p style='font-size: 18px; color: #666;'>AI-Powered Financial Analysis & Predictions with RPT-1-OSS Model by Amit Lal</p>
    </div>
    """)
    
    with gr.Tabs():
        # Dashboard Tab
        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("## 📈 Financial Overview")
            gr.Markdown("*Real-time metrics and key financial indicators*")
            metrics_display = gr.HTML()
            with gr.Row():
                chart1 = gr.Plot()
                chart2 = gr.Plot()
            chart3 = gr.Plot()
            
            refresh_btn = gr.Button("Refresh Dashboard")
            refresh_btn.click(
                create_dashboard,
                outputs=[metrics_display, chart1, chart2, chart3]
            )
            
            # Load dashboard on startup
            app.load(create_dashboard, outputs=[metrics_display, chart1, chart2, chart3])
        
        # Data Explorer Tab
        with gr.TabItem("🔍 Data Explorer"):
            gr.Markdown("## 🗂️ Explore Datasets")
            gr.Markdown("*Browse and analyze your financial data*")
            dataset_selector = gr.Dropdown(
                choices=["GL Accounts", "Financial Statements", "Sales Orders", "Uploaded Data"],
                label="Select Dataset",
                value="GL Accounts"
            )
            info_text = gr.Textbox(label="Dataset Info", interactive=False)
            data_chart = gr.Plot()
            data_table = gr.Dataframe()
            
            dataset_selector.change(
                explore_dataset,
                inputs=[dataset_selector],
                outputs=[info_text, data_chart, data_table]
            )
        
        # Upload Tab
        with gr.TabItem("📤 Upload"):
            gr.Markdown("## 📁 Upload Dataset")
            gr.Markdown("*Upload your own CSV files for analysis*")
            file_upload = gr.File(label="Upload CSV File", file_types=[".csv"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            uploaded_preview = gr.Dataframe()
            
            file_upload.upload(
                upload_file,
                inputs=[file_upload],
                outputs=[upload_status, uploaded_preview]
            )
        
        # Predictions Tab
        with gr.TabItem("🤖 AI Predictions"):
            gr.Markdown("## 🎯 AI Predictions with SAP-RPT-1-OSS")
            gr.Markdown("*Train AI models on financial data and make intelligent predictions powered by deep learning*")
            
            with gr.Row():
                model_type_select = gr.Dropdown(
                    choices=["Classifier", "Regressor"],
                    label="Model Type",
                    value="Classifier",
                    info="Classifier: Categorize data | Regressor: Predict numeric values"
                )
                use_gpu_check = gr.Checkbox(label="Use GPU (requires 80GB memory)", value=False)
                init_btn = gr.Button("Initialize Model", variant="primary")
            
            init_status = gr.Textbox(label="Initialization Status", interactive=False)
            
            gr.Markdown("### Step 1: Train the Model")
            with gr.Row():
                train_dataset_select = gr.Dropdown(
                    choices=["Sales Orders", "GL Accounts", "Financial Statements", "Uploaded Data"],
                    label="Select Training Dataset",
                    value="Sales Orders"
                )
                train_btn = gr.Button("Train Model", variant="primary")
            
            train_status = gr.Textbox(label="Training Status", interactive=False, lines=3)
            
            gr.Markdown("### Step 2: Make Predictions")
            with gr.Row():
                pred_dataset_select = gr.Dropdown(
                    choices=["Sales Orders", "Products", "GL Accounts", "Financial Statements", "Uploaded Data"],
                    label="Select Prediction Dataset",
                    value="Sales Orders",
                    info="Choose which dataset to analyze"
                )
                prediction_scenario = gr.Dropdown(
                    choices=[
                        "High Value Order Classification",
                        "Order Priority Classification",
                        "Customer Segment Classification"
                    ],
                    label="Prediction Scenario",
                    value="High Value Order Classification",
                    info="Scenario updates based on selected dataset"
                )
            
            predict_btn = gr.Button("🎯 Make Predictions", variant="primary", size="lg")
            
            pred_status = gr.Textbox(label="Prediction Results", interactive=False, lines=6)
            
            gr.Markdown("### Prediction Visualizations")
            with gr.Row():
                pred_pie_chart = gr.Plot(label="Distribution")
                pred_bar_chart = gr.Plot(label="Summary")
            with gr.Row():
                pred_gauge_chart = gr.Plot(label="Confidence Score")
            
            gr.Markdown("### Detailed Predictions")
            predictions_table = gr.Dataframe(label="Data with Predictions")
            
            gr.Markdown("""
            **Dataset-Specific Scenarios:**
            
            📦 **Sales Orders:**
            - High Value Order: Premium vs standard orders
            - Order Priority: Urgent vs normal handling
            - Customer Segment: VIP vs regular customers
            
            🛍️ **Products:**
            - Product Performance: Best sellers vs low performers
            - Stock Risk: Items needing reorder
            
            💰 **GL Accounts:**
            - Transaction Risk: Flagged vs normal transactions
            - Account Balance: Above vs below average
            - Expense Category: OpEx vs CapEx
            
            📊 **Financial Statements:**
            - Financial Health: Strong vs weak performance
            - Profitability: High vs low margin periods
            - Growth Trend: Revenue growth vs decline
            """)
            
            init_btn.click(
                init_model,
                inputs=[model_type_select, use_gpu_check],
                outputs=[init_status]
            )
            
            train_btn.click(
                train_model,
                inputs=[train_dataset_select],
                outputs=[train_status]
            )
            
            # Update scenarios when dataset changes
            pred_dataset_select.change(
                update_scenarios,
                inputs=[pred_dataset_select],
                outputs=[prediction_scenario]
            )
            
            predict_btn.click(
                make_predictions,
                inputs=[pred_dataset_select, prediction_scenario],
                outputs=[pred_status, predictions_table, pred_pie_chart, pred_bar_chart, pred_gauge_chart]
            )
        
        # OData Tab
        with gr.TabItem("🔗 OData"):
            gr.Markdown("## 🌐 SAP OData Connection")
            gr.Markdown("*Connect to live SAP systems and fetch real-time data*")
            
            test_conn_btn = gr.Button("Test Connection")
            conn_status = gr.Textbox(label="Connection Status", interactive=False)
            
            with gr.Row():
                entity_select = gr.Dropdown(
                    choices=["Sales Orders", "Products", "Line Items", "Business Partners"],
                    label="Select Entity",
                    value="Sales Orders"
                )
                num_records = gr.Number(label="Number of Records", value=100, minimum=1, maximum=1000)
                fetch_btn = gr.Button("Fetch Data")
            
            fetch_status = gr.Textbox(label="Fetch Status", interactive=False)
            odata_table = gr.Dataframe()
            
            test_conn_btn.click(
                test_odata_connection,
                outputs=[conn_status]
            )
            
            fetch_btn.click(
                fetch_odata_data,
                inputs=[entity_select, num_records],
                outputs=[fetch_status, odata_table]
            )
        
        # Playground Tab
        with gr.TabItem("🎮 Playground"):
            gr.Markdown("## 🧪 SAP-RPT-1-OSS Playground")
            gr.Markdown("*Upload datasets, configure models, and train with real-time progress tracking*")
            
            # Embedding Server Status
            gr.Markdown("**💡 Note:** The SAP-RPT-OSS embedding server starts automatically when the model makes predictions. Manual start is optional and may not be available in all installations.")
            with gr.Row():
                embedding_status_btn = gr.Button("Check Embedding Server", size="sm")
                embedding_status = gr.Textbox(label="Embedding Server Status", interactive=False, lines=4)
                start_server_btn = gr.Button("Start Embedding Server (Optional)", size="sm", variant="secondary")
            
            embedding_status_btn.click(
                check_playground_embedding_server,
                outputs=[embedding_status]
            )
            
            def start_playground_embedding_server():
                """Start embedding server and return formatted message."""
                # Check if package is installed first
                if not is_sap_rpt_oss_installed():
                    return f"❌ sap-rpt-oss package not found\n\n📦 Installation Required:\npip install git+https://github.com/SAP-samples/sap-rpt-1-oss"
                
                success, message = start_embedding_server(None)
                if success:
                    return f"✅ {message}\n\nThe server is now running and will be used automatically during training."
                else:
                    # This is normal - SAP-RPT-OSS starts the server automatically when needed
                    return f"ℹ️ {message}\n\n✅ This is expected! The embedding server will start automatically when you train the model or make predictions. No action needed."
            
            start_server_btn.click(
                start_playground_embedding_server,
                outputs=[embedding_status]
            )
            
            gr.Markdown("### Step 1: Upload Dataset")
            playground_upload = gr.File(
                label="Upload Dataset (CSV, Parquet, or JSON)",
                file_types=[".csv", ".parquet", ".json", ".jsonl"]
            )
            
            playground_info = gr.Textbox(label="Dataset Info", interactive=False, lines=8)
            playground_preview = gr.Dataframe(label="Preview (First 10 Rows)")
            
            gr.Markdown("### Step 2: Configure Model")
            
            # Documentation section
            with gr.Accordion("📚 Parameter Guide - Click to expand", open=False):
                gr.Markdown("""
                **Understanding Model Parameters:**
                
                **🎯 Task Type:**
                - **Classification**: Predicts categories/classes (e.g., "High Risk" vs "Low Risk", "Approved" vs "Rejected")
                  - Target column should have discrete values (integers or categories)
                  - Examples: Will invoice be paid late? (Yes/No), Product category (A/B/C)
                - **Regression**: Predicts continuous numeric values (e.g., price, days, amount)
                  - Target column should have numeric values
                  - Examples: Days until payment, Revenue amount, Risk score (0-100)
                
                **📊 Test Split Ratio:**
                - Proportion of your dataset reserved for testing model performance
                - **0.1 (10%)**: Use more data for training, less for validation. Good for small datasets.
                - **0.2 (20%)**: Balanced approach. Recommended default for most cases.
                - **0.3-0.5 (30-50%)**: More data for testing. Use when you have large datasets and want thorough validation.
                - Higher test split = more reliable performance estimate, but less training data
                
                **🧠 Max Context Size:**
                - Number of examples the model can consider simultaneously when making predictions
                - **512**: Fast, memory-efficient. Good for quick experiments or CPU-only setups.
                - **1024**: Balanced performance. Recommended for most use cases.
                - **2048**: Better accuracy, moderate memory. Good default for production.
                - **4096**: High accuracy, requires significant memory (16GB+ RAM).
                - **8192**: Best accuracy, requires 80GB GPU memory. Use only with powerful hardware.
                - Larger context = better understanding of patterns, but slower and more memory-intensive
                
                **🎲 Bagging Factor:**
                - Number of independent models trained and combined (ensemble learning)
                - **1**: Single model. Fastest, baseline performance.
                - **2**: Two models averaged. Good balance of speed and accuracy. Recommended default.
                - **4**: Four models. Better accuracy, 2x slower than bagging=2.
                - **8**: Eight models. Best accuracy, 4x slower. Use for final production models.
                - Higher bagging = more robust predictions (reduces overfitting), but slower training
                
                **💻 Use GPU:**
                - Enable GPU acceleration (requires NVIDIA GPU with 80GB VRAM)
                - GPU mode: Context size 8192, Bagging 8 (maximum performance)
                - CPU mode: Context size 2048, Bagging 1 (lightweight, works on any machine)
                - Leave unchecked unless you have enterprise-grade GPU hardware
                
                **🔧 Handle Missing Values:**
                - How to treat empty/null values in your data
                - **mean**: Replace with column average (good for normally distributed data)
                - **median**: Replace with column median (better for skewed data, robust to outliers)
                - **zero**: Replace with 0 (simple, but may introduce bias)
                - **drop**: Remove rows with missing values (loses data, but preserves original distribution)
                
                **📏 Normalize Features:**
                - Scale all numeric features to have mean=0 and std=1
                - **Enabled**: Recommended when features have very different scales (e.g., age 0-100 vs income 0-1000000)
                - **Disabled**: Use original feature scales (faster, works when scales are similar)
                - Normalization helps models converge faster and perform better with mixed-scale features
                """)
            
            with gr.Row():
                playground_task_type = gr.Dropdown(
                    choices=["classification", "regression"],
                    label="Task Type",
                    value="classification",
                    info="Classification: Predict categories (Yes/No, A/B/C). Regression: Predict numbers (price, days, score)"
                )
                playground_target_col = gr.Dropdown(
                    choices=[],
                    label="Target Column",
                    value=None,
                    info="The column you want to predict. Auto-selected: last column in dataset"
                )
            
            with gr.Row():
                playground_test_split = gr.Slider(
                    minimum=0.1,
                    maximum=0.5,
                    value=0.2,
                    step=0.05,
                    label="Test Split Ratio",
                    info="Proportion of data for testing (0.2 = 20% test, 80% train). Higher = more validation data, less training data"
                )
                playground_max_context = gr.Dropdown(
                    choices=[512, 1024, 2048, 4096, 8192],
                    value=2048,
                    label="Max Context Size",
                    info="How many examples model considers (512=fast/light, 2048=balanced, 8192=best/needs GPU). Larger = better accuracy, more memory"
                )
            
            with gr.Row():
                playground_bagging = gr.Dropdown(
                    choices=[1, 2, 4, 8],
                    value=2,
                    label="Bagging Factor",
                    info="Number of models to combine (1=fast, 2=balanced, 8=best). Higher = more accurate but slower. Reduces overfitting"
                )
                playground_use_gpu = gr.Checkbox(
                    label="Use GPU (requires 80GB VRAM)",
                    value=False,
                    info="Enable GPU acceleration. Only check if you have NVIDIA GPU with 80GB memory. Unchecked = CPU mode (works on any machine)"
                )
            
            with gr.Row():
                playground_handle_missing = gr.Dropdown(
                    choices=["mean", "median", "zero", "drop"],
                    value="mean",
                    label="Handle Missing Values",
                    info="How to treat empty cells: mean/median (fill with average), zero (fill with 0), drop (remove rows)"
                )
                playground_normalize = gr.Checkbox(
                    label="Normalize Features",
                    value=False,
                    info="Scale all numeric features to same range (mean=0, std=1). Recommended when features have very different scales"
                )
            
            gr.Markdown("### Step 3: Train Model")
            train_playground_btn = gr.Button("🚀 Train Model", variant="primary", size="lg")
            playground_train_status = gr.Textbox(label="Training Status", interactive=False, lines=6)
            
            gr.Markdown("### Step 4: Results")
            playground_results_table = gr.Dataframe(label="Test Predictions (First 100 Rows)")
            playground_download = gr.File(label="Download Full Results CSV")
            
            # Connect upload handler
            def update_playground_components(file):
                """Update all playground components after upload."""
                result = handle_playground_upload(file)
                if len(result) == 7:
                    info, preview, choices, value, task_type, choices2, value2 = result
                    return (
                        info,
                        preview,
                        gr.Dropdown(choices=choices, value=value),
                        task_type,
                        gr.Dropdown(choices=choices2, value=value2)
                    )
                elif len(result) == 7 and result[2] == []:  # Error case
                    return result[0], result[1], gr.Dropdown(choices=[], value=None), result[4], gr.Dropdown(choices=[], value=None)
                return result
            
            playground_upload.upload(
                update_playground_components,
                inputs=[playground_upload],
                outputs=[
                    playground_info,
                    playground_preview,
                    playground_target_col,
                    playground_task_type,
                    playground_target_col
                ]
            )
            
            # Connect training handler
            train_playground_btn.click(
                train_playground_model,
                inputs=[
                    playground_task_type,
                    playground_target_col,
                    playground_test_split,
                    playground_max_context,
                    playground_bagging,
                    playground_use_gpu,
                    playground_handle_missing,
                    playground_normalize
                ],
                outputs=[
                    playground_train_status,
                    playground_results_table,
                    playground_download,
                    playground_download
                ]
            )
            
            with gr.Accordion("💡 Quick Start Guide", open=False):
                gr.Markdown("""
                **Recommended Settings by Use Case:**
                
                **🚀 Quick Experiment (Fast, Low Memory):**
                - Task Type: Auto-detect
                - Test Split: 0.2 (20%)
                - Max Context: 512
                - Bagging: 1
                - GPU: Unchecked
                - Missing Values: mean
                - Normalize: Unchecked
                - *Best for: Trying out the model, small datasets, CPU-only machines*
                
                **⚖️ Balanced (Recommended Default):**
                - Task Type: Auto-detect
                - Test Split: 0.2 (20%)
                - Max Context: 2048
                - Bagging: 2
                - GPU: Unchecked
                - Missing Values: mean
                - Normalize: Check if features have very different scales
                - *Best for: Most production use cases, good accuracy/speed balance*
                
                **🏆 Maximum Accuracy (Slow, High Memory):**
                - Task Type: Auto-detect
                - Test Split: 0.3 (30%)
                - Max Context: 8192
                - Bagging: 8
                - GPU: Checked (requires 80GB GPU)
                - Missing Values: median (more robust)
                - Normalize: Checked
                - *Best for: Final production models, large datasets, when accuracy is critical*
                
                **📋 Step-by-Step Workflow:**
                1. **Upload Dataset**: CSV, Parquet, or JSON file
                2. **Review Auto-Detection**: Check if task type and target column are correct
                3. **Adjust Parameters**: Use recommended settings above or customize
                4. **Train Model**: Click "Train Model" and wait for progress
                5. **Review Results**: Check accuracy/metrics and download predictions
                
                **⚠️ Common Issues:**
                - **"Unknown label type"**: Target column has wrong data type. Change Task Type or convert target column.
                - **Out of Memory**: Reduce Max Context Size or Bagging Factor
                - **Slow Training**: Reduce Bagging Factor or Max Context Size
                - **Poor Accuracy**: Increase Max Context Size, Bagging Factor, or check data quality
                """)
            
            gr.Markdown("""
            **Playground Features:**
            - Upload CSV, Parquet, or JSON datasets
            - Auto-detect task type from filename and target column
            - Auto-select target column (defaults to last column)
            - Configure model parameters with detailed guidance
            - Real-time progress tracking during training
            - Download results as CSV with predictions
            
            **Example Use Cases:**
            - Predictive business outcomes (invoice late payment, days to payment)
            - Recommendations & auto-defaulting (form of address)
            - Normalization & coding (country ISO codes)
            - Data quality & anomaly flags (bank details review)
            - Derived scores & segments (employee risk of leave)
            - Matching & linking (material entity matching)
            - Information extraction (ticket topic classification)
            """)


if __name__ == "__main__":
    import os
    
    # Load datasets on startup
    load_datasets()
    
    # Get server configuration from environment variables (for container deployment)
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", 7862))
    
    # Launch the app
    app.launch(share=False, server_name=server_name, server_port=server_port, quiet=False)

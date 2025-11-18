"""
SAP Finance Dashboard with RPT-1-OSS Model

Main Mesop application with multiple pages:
- Dashboard: Overview with metrics and charts
- Data Explorer: Browse datasets
- Upload: Upload custom datasets
- Predictions: AI-powered predictions using SAP-RPT-1-OSS
- OData: Connect to SAP OData services
"""

import os
import mesop as me
import pandas as pd
import numpy as np
from pathlib import Path
import json
import base64
from typing import Optional, Dict, Any
import plotly.graph_objects as go
import plotly.io as pio

# Import utilities
from utils.data_generator import generate_all_datasets
from utils.visualizations import (
    create_revenue_expense_chart,
    create_balance_sheet_chart,
    create_gl_summary_chart,
    create_sales_analytics_chart,
    create_sales_trend_chart,
    get_summary_metrics
)
from utils.odata_connector import SAPFinanceConnector
from models.rpt_model import RPTModelWrapper, create_model

# Global state
from dataclasses import field

@me.stateclass
class State:
    gl_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    financial_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    sales_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    uploaded_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_dataset_type: str = ""
    odata_connector: Optional[SAPFinanceConnector] = None
    odata_connected: bool = False
    odata_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_wrapper: Optional[RPTModelWrapper] = None
    predictions: Optional[np.ndarray] = None
    prediction_proba: Optional[np.ndarray] = None
    connection_message: str = ""
    fetch_message: str = ""
    model_initialized: bool = False
    model_trained: bool = False


def load_datasets(state: State):
    """Load synthetic datasets if they exist."""
    data_dir = Path("data")
    
    if (data_dir / "synthetic_gl_accounts.csv").exists():
        state.gl_data = pd.read_csv(data_dir / "synthetic_gl_accounts.csv")
    
    if (data_dir / "synthetic_financial_statements.csv").exists():
        state.financial_data = pd.read_csv(data_dir / "synthetic_financial_statements.csv")
    
    if (data_dir / "synthetic_sales_orders.csv").exists():
        state.sales_data = pd.read_csv(data_dir / "synthetic_sales_orders.csv")


def plotly_to_html(fig_dict: Dict[str, Any]) -> str:
    """Convert Plotly figure dict to HTML string."""
    if not fig_dict:
        return "<p>No chart data available</p>"
    
    try:
        fig = go.Figure(fig_dict)
        html_str = pio.to_html(fig, include_plotlyjs='cdn', div_id="plotly-div")
        return html_str
    except Exception as e:
        return f"<p>Error rendering chart: {str(e)}</p>"


@me.page(path="/", title="SAP Finance Dashboard")
def dashboard_page():
    """Main dashboard page with overview metrics and charts."""
    state = me.state(State)
    
    me.text("SAP Finance Dashboard", style=me.Style(font_size=32, font_weight="bold", margin=me.Margin(bottom=16)))
    
    # Load datasets if not loaded
    if state.gl_data.empty and state.financial_data.empty and state.sales_data.empty:
        load_datasets(state)
    
    # Generate datasets if they don't exist
    if state.gl_data.empty or state.financial_data.empty or state.sales_data.empty:
        with me.box(style=me.Style(padding=16, background="#fff3cd", border_radius=8, margin=me.Margin(bottom=16))):
            me.text("Generating synthetic datasets...", style=me.Style(color="#856404"))
            generate_all_datasets()
            load_datasets(state)
    
    # Summary metrics
    with me.box(style=me.Style(display="grid", grid_template_columns="repeat(4, 1fr)", gap=16, margin=me.Margin(bottom=24))):
        if not state.gl_data.empty:
            gl_metrics = get_summary_metrics(state.gl_data, "gl")
            with me.box(style=me.Style(padding=16, background="#f8f9fa", border_radius=8)):
                me.text("GL Transactions", style=me.Style(font_weight="bold"))
                me.text(f"{gl_metrics.get('Total Transactions', 0):,}")
        
        if not state.financial_data.empty:
            fin_metrics = get_summary_metrics(state.financial_data, "financial")
            with me.box(style=me.Style(padding=16, background="#f8f9fa", border_radius=8)):
                me.text("Latest Revenue", style=me.Style(font_weight="bold"))
                me.text(f"${fin_metrics.get('Latest Revenue', 0):,.2f}")
        
        if not state.sales_data.empty:
            sales_metrics = get_summary_metrics(state.sales_data, "sales")
            with me.box(style=me.Style(padding=16, background="#f8f9fa", border_radius=8)):
                me.text("Total Sales", style=me.Style(font_weight="bold"))
                me.text(f"${sales_metrics.get('Total Sales', 0):,.2f}")
        
        with me.box(style=me.Style(padding=16, background="#f8f9fa", border_radius=8)):
            me.text("Datasets", style=me.Style(font_weight="bold"))
            count = sum([
                not state.gl_data.empty,
                not state.financial_data.empty,
                not state.sales_data.empty,
                not state.uploaded_data.empty
            ])
            me.text(f"{count} loaded")
    
    # Charts
    if not state.financial_data.empty:
        with me.box(style=me.Style(margin=me.Margin(bottom=24))):
            me.text("Financial Trends", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
            chart_data = create_revenue_expense_chart(state.financial_data)
            if chart_data:
                html_chart = plotly_to_html(chart_data)
                me.html(html_chart)
    
    if not state.financial_data.empty:
        with me.box(style=me.Style(margin=me.Margin(bottom=24))):
            me.text("Balance Sheet", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
            chart_data = create_balance_sheet_chart(state.financial_data)
            if chart_data:
                html_chart = plotly_to_html(chart_data)
                me.html(html_chart)
    
    if not state.sales_data.empty:
        with me.box(style=me.Style(margin=me.Margin(bottom=24))):
            me.text("Sales Analytics", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
            chart_data = create_sales_analytics_chart(state.sales_data)
            if chart_data:
                html_chart = plotly_to_html(chart_data)
                me.html(html_chart)


@me.page(path="/explorer", title="Data Explorer")
def explorer_page():
    """Data explorer page to browse and filter datasets."""
    state = me.state(State)
    
    me.text("Data Explorer", style=me.Style(font_size=32, font_weight="bold", margin=me.Margin(bottom=16)))
    
    # Dataset selector
    with me.box(style=me.Style(margin=me.Margin(bottom=16))):
        me.text("Select Dataset:", style=me.Style(font_weight="bold", margin=me.Margin(bottom=8)))
        dataset_options = [
            ("GL Accounts", "gl"),
            ("Financial Statements", "financial"),
            ("Sales Orders", "sales"),
            ("Uploaded Data", "uploaded")
        ]
        
        me.select(
            label="Dataset",
            options=[me.SelectOption(label=label, value=value) for label, value in dataset_options],
            on_selection_change=on_dataset_selection_change,
            value=state.current_dataset_type
        )
    
    # Display selected dataset
    if state.current_dataset_type:
        display_dataset(state, state.current_dataset_type)


def on_dataset_selection_change(e: me.SelectSelectionChangeEvent):
    """Handle dataset selection change."""
    state = me.state(State)
    state.current_dataset_type = e.value


def display_dataset(state: State, dataset_type: str):
    """Display the selected dataset."""
    if dataset_type == "gl" and not state.gl_data.empty:
        df = state.gl_data
        me.text(f"GL Accounts ({len(df)} records)", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
        chart_data = create_gl_summary_chart(df)
        if chart_data:
            html_chart = plotly_to_html(chart_data)
            me.html(html_chart)
        me.table(data=df.head(100).to_dict("records"), style=me.Style(margin=me.Margin(top=16)))
    
    elif dataset_type == "financial" and not state.financial_data.empty:
        df = state.financial_data
        me.text(f"Financial Statements ({len(df)} records)", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
        chart_data = create_revenue_expense_chart(df)
        if chart_data:
            html_chart = plotly_to_html(chart_data)
            me.html(html_chart)
        me.table(data=df.to_dict("records"), style=me.Style(margin=me.Margin(top=16)))
    
    elif dataset_type == "sales" and not state.sales_data.empty:
        df = state.sales_data
        me.text(f"Sales Orders ({len(df)} records)", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
        chart_data = create_sales_trend_chart(df)
        if chart_data:
            html_chart = plotly_to_html(chart_data)
            me.html(html_chart)
        me.table(data=df.head(100).to_dict("records"), style=me.Style(margin=me.Margin(top=16)))
    
    elif dataset_type == "uploaded" and not state.uploaded_data.empty:
        df = state.uploaded_data
        me.text(f"Uploaded Data ({len(df)} records)", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
        me.table(data=df.head(100).to_dict("records"), style=me.Style(margin=me.Margin(top=16)))
    
    else:
        me.text("No data available for this dataset type.", style=me.Style(color="#dc3545"))


@me.page(path="/upload", title="Upload Data")
def upload_page():
    """Upload page for custom datasets."""
    state = me.state(State)
    
    me.text("Upload Dataset", style=me.Style(font_size=32, font_weight="bold", margin=me.Margin(bottom=16)))
    
    with me.box(style=me.Style(margin=me.Margin(bottom=16))):
        me.text("Upload a CSV file to analyze:", style=me.Style(margin=me.Margin(bottom=8)))
        me.file_upload(
            label="Choose CSV File",
            accept=".csv",
            on_upload=handle_file_upload
        )
    
    if not state.uploaded_data.empty:
        me.text("Uploaded Data Preview:", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(top=16, bottom=8)))
        me.table(data=state.uploaded_data.head(50).to_dict("records"))


def handle_file_upload(e: me.UploadEvent):
    """Handle file upload."""
    state = me.state(State)
    try:
        if e.files:
            file = e.files[0]
            df = pd.read_csv(file.getvalue())
            state.uploaded_data = df
    except Exception as ex:
        pass


@me.page(path="/predictions", title="Predictions")
def predictions_page():
    """Predictions page using SAP-RPT-1-OSS model."""
    state = me.state(State)
    
    me.text("AI Predictions with SAP-RPT-1-OSS", style=me.Style(font_size=32, font_weight="bold", margin=me.Margin(bottom=16)))
    
    with me.box(style=me.Style(margin=me.Margin(bottom=16))):
        me.text("Model Configuration", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
        
        me.select(
            label="Model Type",
            options=[
                me.SelectOption(label="Classifier", value="classifier"),
                me.SelectOption(label="Regressor", value="regressor")
            ]
        )
        
        me.checkbox(label="Use GPU (requires 80GB memory)", checked=False)
        
        me.button("Initialize Model", on_click=on_init_model)
    
    if state.model_initialized:
        me.text("Model initialized successfully!", style=me.Style(color="#28a745", margin=me.Margin(bottom=16)))
        
        # Dataset selection for training
        with me.box(style=me.Style(margin=me.Margin(bottom=16))):
            me.text("Select Training Data", style=me.Style(font_weight="bold", margin=me.Margin(bottom=8)))
            dataset_options = [
                ("GL Accounts", "gl"),
                ("Financial Statements", "financial"),
                ("Sales Orders", "sales"),
                ("Uploaded Data", "uploaded")
            ]
            
            me.select(
                label="Dataset",
                options=[me.SelectOption(label=label, value=value) for label, value in dataset_options]
            )
        
        me.button("Train Model", on_click=on_train_model)
        
        if state.model_trained:
            me.text("Model trained successfully!", style=me.Style(color="#28a745", margin=me.Margin(top=16)))
        
        if state.predictions is not None:
            me.text("Predictions:", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(top=16, bottom=8)))
            me.text(str(state.predictions[:10]))  # Show first 10 predictions
    
    else:
        with me.box(style=me.Style(padding=16, background="#fff3cd", border_radius=8)):
            me.text("Please initialize the model first.", style=me.Style(color="#856404"))


def on_init_model(e: me.ClickEvent):
    """Initialize the model."""
    state = me.state(State)
    try:
        state.model_wrapper = create_model(model_type="classifier", use_gpu=False)
        state.model_initialized = True
    except Exception as ex:
        state.connection_message = f"Error initializing model: {str(ex)}"


def on_train_model(e: me.ClickEvent):
    """Train the model."""
    state = me.state(State)
    try:
        if state.model_wrapper and not state.gl_data.empty:
            # Simple example: use GL data
            X = state.gl_data.select_dtypes(include=[np.number]).dropna()
            if len(X) > 0:
                # Create a simple target for classification
                y = (X.iloc[:, 0] > X.iloc[:, 0].median()).astype(int)
                state.model_wrapper.fit(X, y)
                state.model_trained = True
    except Exception as ex:
        state.connection_message = f"Error training model: {str(ex)}"


@me.page(path="/odata", title="OData Connection")
def odata_page():
    """OData connection page for SAP data."""
    state = me.state(State)
    
    me.text("SAP OData Connection", style=me.Style(font_size=32, font_weight="bold", margin=me.Margin(bottom=16)))
    
    # Connection status
    with me.box(style=me.Style(margin=me.Margin(bottom=16))):
        if state.odata_connector is None:
            state.odata_connector = SAPFinanceConnector()
        
        me.button("Test Connection", on_click=on_test_odata_connection)
        
        if state.connection_message:
            color = "#28a745" if state.odata_connected else "#dc3545"
            me.text(state.connection_message, style=me.Style(color=color, margin=me.Margin(top=8)))
        elif state.odata_connected:
            me.text("✓ Connected to SAP OData", style=me.Style(color="#28a745", margin=me.Margin(top=8)))
        else:
            me.text("Not connected", style=me.Style(color="#dc3545", margin=me.Margin(top=8)))
    
    # Fetch options
    if state.odata_connected:
        with me.box(style=me.Style(margin=me.Margin(bottom=16))):
            me.text("Fetch Data", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(bottom=8)))
            
            top_count = me.number_input(label="Number of records", value=100, min_value=1, max_value=1000)
            
            with me.box(style=me.Style(display="grid", grid_template_columns="repeat(2, 1fr)", gap=8, margin=me.Margin(top=8))):
                me.button("Fetch Sales Orders", on_click=lambda e: on_fetch_odata(e, "orders", top_count))
                me.button("Fetch Products", on_click=lambda e: on_fetch_odata(e, "products", top_count))
                me.button("Fetch Line Items", on_click=lambda e: on_fetch_odata(e, "line_items", top_count))
                me.button("Fetch Partners", on_click=lambda e: on_fetch_odata(e, "partners", top_count))
        
        if state.fetch_message:
            me.text(state.fetch_message, style=me.Style(color="#28a745", margin=me.Margin(top=8)))
        
        # Display fetched data
        if not state.odata_data.empty:
            me.text("Fetched Data:", style=me.Style(font_size=20, font_weight="bold", margin=me.Margin(top=16, bottom=8)))
            me.table(data=state.odata_data.head(100).to_dict("records"))


def on_test_odata_connection(e: me.ClickEvent):
    """Test OData connection."""
    state = me.state(State)
    try:
        if state.odata_connector is None:
            state.odata_connector = SAPFinanceConnector()
        
        connected, message = state.odata_connector.test_connection()
        state.odata_connected = connected
        state.connection_message = message
    except Exception as ex:
        state.connection_message = f"Error: {str(ex)}"
        state.odata_connected = False


def on_fetch_odata(e: me.ClickEvent, entity_type: str, top: int):
    """Fetch data from OData service."""
    state = me.state(State)
    try:
        if not state.odata_connected:
            state.fetch_message = "Please connect first!"
            return
        
        if entity_type == "orders":
            state.odata_data = state.odata_connector.fetch_orders_df(top)
        elif entity_type == "products":
            state.odata_data = state.odata_connector.fetch_products_df(top)
        elif entity_type == "line_items":
            state.odata_data = state.odata_connector.fetch_line_items_df(top)
        elif entity_type == "partners":
            state.odata_data = state.odata_connector.fetch_partners_df(top)
        
        state.fetch_message = f"Fetched {len(state.odata_data)} records"
    except Exception as ex:
        state.fetch_message = f"Error fetching data: {str(ex)}"


# Navigation
@me.page(path="/nav", title="Navigation")
def nav_page():
    """Navigation page."""
    me.text("Navigation", style=me.Style(font_size=32, font_weight="bold", margin=me.Margin(bottom=16)))
    
    nav_links = [
        ("Dashboard", "/"),
        ("Data Explorer", "/explorer"),
        ("Upload", "/upload"),
        ("Predictions", "/predictions"),
        ("OData", "/odata")
    ]
    
    for label, path in nav_links:
        with me.box(style=me.Style(margin=me.Margin(bottom=8))):
            me.link(label, path=path, style=me.Style(font_size=18, text_decoration="none"))


if __name__ == "__main__":
    # Generate datasets if they don't exist
    data_dir = Path("data")
    if not (data_dir / "synthetic_gl_accounts.csv").exists():
        generate_all_datasets()
    
    me.run()


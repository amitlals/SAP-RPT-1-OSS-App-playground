"""
Visualization Utilities for Financial Data

Creates Plotly charts for:
- Revenue/Expense trends
- Balance Sheet visualizations
- GL account transaction summaries
- Sales order analytics
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, Dict, Any


def create_revenue_expense_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a revenue and expense trend chart.
    
    Args:
        df: DataFrame with financial statement data
        
    Returns:
        Plotly figure as dictionary (JSON-serializable)
    """
    if df.empty or 'Period' not in df.columns:
        return {}
    
    fig = go.Figure()
    
    if 'Revenue' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Period'],
            y=df['Revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8)
        ))
    
    if 'Operating_Expenses' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Period'],
            y=df['Operating_Expenses'],
            mode='lines+markers',
            name='Operating Expenses',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
    
    if 'Net_Income' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Period'],
            y=df['Net_Income'],
            mode='lines+markers',
            name='Net Income',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='Revenue and Expense Trends',
        xaxis_title='Period',
        yaxis_title='Amount (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig.to_dict()


def create_balance_sheet_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a balance sheet visualization.
    
    Args:
        df: DataFrame with balance sheet data
        
    Returns:
        Plotly figure as dictionary (JSON-serializable)
    """
    if df.empty:
        return {}
    
    # Get the most recent period
    latest = df.iloc[-1] if len(df) > 0 else df.iloc[0]
    
    # Assets
    assets = {
        'Cash': latest.get('Cash', 0),
        'Accounts Receivable': latest.get('Accounts_Receivable', 0),
        'Inventory': latest.get('Inventory', 0),
        'PPE': latest.get('PPE', 0)
    }
    
    # Liabilities
    liabilities = {
        'Accounts Payable': latest.get('Accounts_Payable', 0),
        'Short-term Debt': latest.get('Short_Term_Debt', 0),
        'Long-term Debt': latest.get('Long_Term_Debt', 0)
    }
    
    # Equity
    equity = {
        'Equity': latest.get('Equity', 0)
    }
    
    fig = go.Figure()
    
    # Assets bar
    fig.add_trace(go.Bar(
        name='Assets',
        x=list(assets.keys()),
        y=list(assets.values()),
        marker_color='#2ecc71'
    ))
    
    # Liabilities bar
    fig.add_trace(go.Bar(
        name='Liabilities',
        x=list(liabilities.keys()),
        y=list(liabilities.values()),
        marker_color='#e74c3c'
    ))
    
    # Equity bar
    fig.add_trace(go.Bar(
        name='Equity',
        x=list(equity.keys()),
        y=list(equity.values()),
        marker_color='#3498db'
    ))
    
    fig.update_layout(
        title='Balance Sheet Overview',
        xaxis_title='Category',
        yaxis_title='Amount (USD)',
        barmode='group',
        template='plotly_white',
        height=400
    )
    
    return fig.to_dict()


def create_gl_summary_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a GL account transaction summary chart.
    
    Args:
        df: DataFrame with GL transaction data
        
    Returns:
        Plotly figure as dictionary (JSON-serializable)
    """
    if df.empty or 'Account_Description' not in df.columns:
        return {}
    
    # Aggregate by account
    account_summary = df.groupby('Account_Description').agg({
        'Debit': 'sum',
        'Credit': 'sum'
    }).reset_index()
    
    account_summary['Net'] = account_summary['Debit'] - account_summary['Credit']
    account_summary = account_summary.sort_values('Net', ascending=True).tail(15)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=account_summary['Account_Description'],
        x=account_summary['Net'],
        orientation='h',
        marker=dict(
            color=account_summary['Net'],
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title='Top 15 GL Accounts by Net Balance',
        xaxis_title='Net Balance (USD)',
        yaxis_title='Account',
        template='plotly_white',
        height=500
    )
    
    return fig.to_dict()


def create_sales_analytics_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create sales order analytics chart.
    
    Args:
        df: DataFrame with sales order data
        
    Returns:
        Plotly figure as dictionary (JSON-serializable)
    """
    if df.empty:
        return {}
    
    fig = go.Figure()
    
    # Sales by region
    if 'Region' in df.columns and 'Total_Amount' in df.columns:
        region_sales = df.groupby('Region')['Total_Amount'].sum().reset_index()
        
        fig.add_trace(go.Bar(
            x=region_sales['Region'],
            y=region_sales['Total_Amount'],
            marker_color='#3498db',
            text=region_sales['Total_Amount'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Sales by Region',
            xaxis_title='Region',
            yaxis_title='Total Sales (USD)',
            template='plotly_white',
            height=400
        )
    elif 'Product_Name' in df.columns and 'Total_Amount' in df.columns:
        # Sales by product
        product_sales = df.groupby('Product_Name')['Total_Amount'].sum().reset_index()
        product_sales = product_sales.sort_values('Total_Amount', ascending=False).head(10)
        
        fig.add_trace(go.Bar(
            x=product_sales['Product_Name'],
            y=product_sales['Total_Amount'],
            marker_color='#9b59b6',
            text=product_sales['Total_Amount'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Top 10 Products by Sales',
            xaxis_title='Product',
            yaxis_title='Total Sales (USD)',
            template='plotly_white',
            height=400,
            xaxis_tickangle=-45
        )
    
    return fig.to_dict()


def create_sales_trend_chart(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a sales trend over time chart.
    
    Args:
        df: DataFrame with sales order data
        
    Returns:
        Plotly figure as dictionary (JSON-serializable)
    """
    if df.empty or 'Order_Date' not in df.columns:
        return {}
    
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    df['Month'] = df['Order_Date'].dt.to_period('M').astype(str)
    
    monthly_sales = df.groupby('Month')['Total_Amount'].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_sales['Month'],
        y=monthly_sales['Total_Amount'],
        mode='lines+markers',
        name='Monthly Sales',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8),
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.update_layout(
        title='Sales Trend Over Time',
        xaxis_title='Month',
        yaxis_title='Total Sales (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig.to_dict()


def create_pie_chart(df: pd.DataFrame, column: str, title: str) -> Dict[str, Any]:
    """
    Create a pie chart for categorical data.
    
    Args:
        df: DataFrame with data
        column: Column name to aggregate
        title: Chart title
        
    Returns:
        Plotly figure as dictionary (JSON-serializable)
    """
    if df.empty or column not in df.columns:
        return {}
    
    value_counts = df[column].value_counts().head(10)
    
    fig = go.Figure(data=[go.Pie(
        labels=value_counts.index,
        values=value_counts.values,
        hole=0.3
    )])
    
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=400
    )
    
    return fig.to_dict()


def get_summary_metrics(df: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
    """
    Get summary metrics for a dataset.
    
    Args:
        df: DataFrame with data
        dataset_type: Type of dataset ('gl', 'financial', 'sales')
        
    Returns:
        Dictionary with summary metrics
    """
    if df.empty:
        return {}
    
    metrics = {}
    
    if dataset_type == 'gl':
        metrics = {
            'Total Transactions': len(df),
            'Total Debit': df['Debit'].sum() if 'Debit' in df.columns else 0,
            'Total Credit': df['Credit'].sum() if 'Credit' in df.columns else 0,
            'Unique Accounts': df['Account_Code'].nunique() if 'Account_Code' in df.columns else 0
        }
    elif dataset_type == 'financial':
        latest = df.iloc[-1] if len(df) > 0 else df.iloc[0]
        metrics = {
            'Periods': len(df),
            'Latest Revenue': latest.get('Revenue', 0),
            'Latest Net Income': latest.get('Net_Income', 0),
            'Total Assets': latest.get('Total_Assets', 0)
        }
    elif dataset_type == 'sales':
        metrics = {
            'Total Orders': len(df),
            'Total Sales': df['Total_Amount'].sum() if 'Total_Amount' in df.columns else 0,
            'Average Order Value': df['Total_Amount'].mean() if 'Total_Amount' in df.columns else 0,
            'Unique Customers': df['Customer_ID'].nunique() if 'Customer_ID' in df.columns else 0
        }
    
    return metrics


def create_prediction_distribution_chart(predictions: list, labels: dict, title: str = "Prediction Distribution") -> Dict[str, Any]:
    """
    Create a pie chart showing prediction distribution.
    
    Args:
        predictions: List of prediction values (0, 1, etc.)
        labels: Dictionary mapping values to labels
        title: Chart title
        
    Returns:
        Plotly figure as dictionary
    """
    import numpy as np
    
    predictions = np.array(predictions)
    unique, counts = np.unique(predictions, return_counts=True)
    
    pie_labels = [labels.get(int(val), f"Class {int(val)}") for val in unique]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    
    fig = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=counts,
        hole=0.4,
        marker=dict(colors=colors[:len(unique)]),
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig.to_dict()


def create_prediction_bar_chart(predictions: list, labels: dict, title: str = "Prediction Summary") -> Dict[str, Any]:
    """
    Create a bar chart showing prediction counts.
    
    Args:
        predictions: List of prediction values
        labels: Dictionary mapping values to labels
        title: Chart title
        
    Returns:
        Plotly figure as dictionary
    """
    import numpy as np
    
    predictions = np.array(predictions)
    unique, counts = np.unique(predictions, return_counts=True)
    
    bar_labels = [labels.get(int(val), f"Class {int(val)}") for val in unique]
    percentages = (counts / len(predictions) * 100).round(1)
    
    colors = ['#3498db' if val == 0 else '#2ecc71' for val in unique]
    
    fig = go.Figure(data=[go.Bar(
        x=bar_labels,
        y=counts,
        marker_color=colors,
        text=[f'{count}<br>({pct}%)' for count, pct in zip(counts, percentages)],
        textposition='outside',
        textfont=dict(size=14, color='#2c3e50')
    )])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#2c3e50')),
        xaxis_title="Classification",
        yaxis_title="Count",
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    return fig.to_dict()


def create_confidence_gauge(confidence_score: float, title: str = "Model Confidence") -> Dict[str, Any]:
    """
    Create a gauge chart showing model confidence.
    
    Args:
        confidence_score: Confidence score (0-100)
        title: Chart title
        
    Returns:
        Plotly figure as dictionary
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': '#e74c3c'},
                {'range': [33, 66], 'color': '#f39c12'},
                {'range': [66, 100], 'color': '#2ecc71'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig.to_dict()


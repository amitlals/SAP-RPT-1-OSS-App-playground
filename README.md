# SAP OpenSource Model Playground | SAP-RPT-1-OSS Model Beta

A comprehensive SAP Playground application built with Gradio that integrates the SAP-RPT-1-OSS model for predictive analysis on SAP datasets. Features include synthetic data generation, interactive visualizations, live OData connectivity, AI-powered insights, and a playground for training custom models. 
<img width="839" height="540" alt="image" src="https://github.com/user-attachments/assets/d6376079-2b83-422a-b4fd-51d517431ec2" />

## Features

- **Multiple Synthetic SAP Datasets**: General Ledger accounts, Financial Statements (P&L, Balance Sheet), and Sales Orders
- **Data Upload**: Upload custom CSV, Parquet, or JSON datasets for analysis
- **Interactive Visualizations**: Financial charts and graphs using Plotly
- **SAP-RPT-1-OSS Model Integration**: AI-powered predictions and analysis
- **Live OData Connection**: Connect to SAP systems to fetch real-time sales order data
- **Playground Tab**: Upload datasets, configure model parameters, train, and download predictions
- **Modern UI**: Built with Gradio, a Python-based web framework

## Installation or HuggingFace (whatever you like!)
https://huggingface.co/spaces/amitgpt/sap-finance-dashboard-RPT-1-OSS 

### Prerequisites

- Python 3.11 or higher
- Hugging Face account (for SAP model access)
- SAP OData credentials (optional, for live data connection)

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SAP-RPT-1-OSS-App
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install SAP-RPT-OSS package**:
   ```bash
   pip install git+https://github.com/SAP-samples/sap-rpt-1-oss
   ```

5. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Fill in your SAP OData credentials
   - Add your Hugging Face token for model access

6. **Authenticate with Hugging Face**:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   Or set the `HUGGINGFACE_TOKEN` in your `.env` file.

7. **Generate synthetic data** (optional, if not already generated):
   ```bash
   python -c "from utils.data_generator import generate_all_datasets; generate_all_datasets()"
   ```

## Usage

### Running the Application

Start the Gradio application:

```bash
python app_gradio.py
```

The application will be available at `http://localhost:7862` (default Gradio port).

### Application Tabs

1. **Dashboard**: Overview with key financial metrics and visualizations
2. **Data Explorer**: Browse and filter datasets (GL, Financial Statements, Sales Orders)
3. **Upload**: Upload custom CSV datasets for analysis
4. **Predictions**: Use SAP-RPT-1-OSS model for predictions and analysis with pre-configured scenarios
5. **OData**: Connect to SAP OData services and fetch live data
6. **Playground**: Upload datasets, configure model parameters (task type, target column, test split, context size, bagging, GPU), train models, and download predictions

## SAP OData Connection Setup

1. Set the following environment variables in your `.env` file:
   - `SAP_USERNAME`: Your SAP username
   - `SAP_PASSWORD`: Your SAP password
   - `SAP_SERVER`: SAP server URL (default: `https://sapes5.sapdevcenter.com/`)
   - `SAP_CLIENT`: SAP client number (default: `002`)

2. The OData connector uses the base URL:
   `https://sapes5.sapdevcenter.com/sap/opu/odata/IWBEP/GWSAMPLE_BASIC`

3. Available endpoints:
   - Sales Orders: `SalesOrderSet`
   - Products: `ProductSet`
   - Line Items: `SalesOrderLineItemSet`
   - Business Partners: `BusinessPartnerSet`

## Model Configuration

The SAP-RPT-1-OSS model supports both classification and regression tasks. For best performance:

- **Recommended**: GPU with at least 80 GB memory, context size 8192, bagging factor 8
- **Lightweight**: CPU with context size 2048, bagging factor 1

The application automatically detects available resources and adjusts settings accordingly.

## Project Structure

```
SAP-RPT-1-OSS-App/
├── app_gradio.py         # Main Gradio application
├── models/
│   └── rpt_model.py      # SAP-RPT-1-OSS model wrapper
├── data/
│   ├── synthetic_gl_accounts.csv
│   ├── synthetic_financial_statements.csv
│   └── synthetic_sales_orders.csv
├── utils/
│   ├── data_generator.py # Generate synthetic SAP finance data
│   ├── visualizations.py # Chart generation functions
│   ├── odata_connector.py # OData connection utilities
│   └── playground.py     # Playground utilities for model training
├── requirements.txt
├── README.md
└── .env.example
```

## License

This project is licensed under the Apache Software License, version 2.0.

## Support

For issues or questions, please create an issue in this repository.

## Acknowledgments

- SAP-RPT-1-OSS model: [Hugging Face](https://huggingface.co/SAP/sap-rpt-1-oss)
- Gradio framework: [Gradio Documentation](https://www.gradio.app/docs/)


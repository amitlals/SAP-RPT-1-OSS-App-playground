"""
Synthetic SAP Finance Data Generator

Generates synthetic datasets for:
- General Ledger accounts with transactions
- Financial Statements (P&L and Balance Sheet)
- Sales Order data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_gl_accounts(num_transactions=1000, output_path="data/synthetic_gl_accounts.csv"):
    """Generate synthetic General Ledger accounts with transactions."""
    
    # Account codes and descriptions
    account_codes = [
        "100000", "110000", "120000", "130000", "140000",  # Assets
        "200000", "210000", "220000", "230000",  # Liabilities
        "300000", "310000", "320000",  # Equity
        "400000", "410000", "420000", "430000",  # Revenue
        "500000", "510000", "520000", "530000", "540000",  # Expenses
    ]
    
    account_descriptions = [
        "Cash and Cash Equivalents", "Accounts Receivable", "Inventory", 
        "Prepaid Expenses", "Property, Plant & Equipment",
        "Accounts Payable", "Accrued Liabilities", "Short-term Debt", "Long-term Debt",
        "Common Stock", "Retained Earnings", "Other Equity",
        "Sales Revenue", "Service Revenue", "Interest Income", "Other Income",
        "Cost of Goods Sold", "Salaries and Wages", "Rent Expense", 
        "Utilities Expense", "Marketing Expense"
    ]
    
    np.random.seed(42)
    
    transactions = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(num_transactions):
        account_idx = np.random.randint(0, len(account_codes))
        transaction_date = base_date + timedelta(days=np.random.randint(0, 365))
        
        # Determine if debit or credit based on account type
        account_num = int(account_codes[account_idx][0])
        if account_num in [1, 5]:  # Assets or Expenses
            debit = np.random.uniform(100, 50000)
            credit = 0
        else:  # Liabilities, Equity, Revenue
            debit = 0
            credit = np.random.uniform(100, 50000)
        
        transactions.append({
            "Transaction_ID": f"TXN{str(i+1).zfill(6)}",
            "Date": transaction_date.strftime("%Y-%m-%d"),
            "Account_Code": account_codes[account_idx],
            "Account_Description": account_descriptions[account_idx],
            "Debit": round(debit, 2),
            "Credit": round(credit, 2),
            "Balance": round(debit - credit, 2),
            "Document_Number": f"DOC{str(np.random.randint(1000, 9999))}",
            "Posting_Period": transaction_date.strftime("%Y-%m")
        })
    
    df = pd.DataFrame(transactions)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_transactions} GL transactions: {output_path}")
    return df


def generate_financial_statements(num_periods=12, output_path="data/synthetic_financial_statements.csv"):
    """Generate synthetic Financial Statements (P&L and Balance Sheet)."""
    
    np.random.seed(42)
    base_date = datetime(2024, 1, 1)
    
    statements = []
    
    for period in range(num_periods):
        period_date = base_date + timedelta(days=period * 30)
        period_str = period_date.strftime("%Y-%m")
        
        # Profit & Loss Statement
        revenue = np.random.uniform(500000, 1000000)
        cogs = revenue * np.random.uniform(0.4, 0.6)
        gross_profit = revenue - cogs
        
        operating_expenses = np.random.uniform(200000, 400000)
        ebitda = gross_profit - operating_expenses
        depreciation = np.random.uniform(20000, 50000)
        ebit = ebitda - depreciation
        interest_expense = np.random.uniform(10000, 30000)
        ebt = ebit - interest_expense
        tax = ebt * 0.25
        net_income = ebt - tax
        
        # Balance Sheet
        cash = np.random.uniform(100000, 500000)
        accounts_receivable = np.random.uniform(200000, 400000)
        inventory = np.random.uniform(150000, 300000)
        current_assets = cash + accounts_receivable + inventory
        ppe = np.random.uniform(2000000, 5000000)
        total_assets = current_assets + ppe
        
        accounts_payable = np.random.uniform(100000, 200000)
        short_term_debt = np.random.uniform(50000, 150000)
        current_liabilities = accounts_payable + short_term_debt
        long_term_debt = np.random.uniform(1000000, 2000000)
        total_liabilities = current_liabilities + long_term_debt
        
        equity = total_assets - total_liabilities
        
        statements.append({
            "Period": period_str,
            "Statement_Type": "P&L",
            "Revenue": round(revenue, 2),
            "Cost_of_Goods_Sold": round(cogs, 2),
            "Gross_Profit": round(gross_profit, 2),
            "Operating_Expenses": round(operating_expenses, 2),
            "EBITDA": round(ebitda, 2),
            "Depreciation": round(depreciation, 2),
            "EBIT": round(ebit, 2),
            "Interest_Expense": round(interest_expense, 2),
            "EBT": round(ebt, 2),
            "Tax": round(tax, 2),
            "Net_Income": round(net_income, 2),
            "Cash": round(cash, 2),
            "Accounts_Receivable": round(accounts_receivable, 2),
            "Inventory": round(inventory, 2),
            "Current_Assets": round(current_assets, 2),
            "PPE": round(ppe, 2),
            "Total_Assets": round(total_assets, 2),
            "Accounts_Payable": round(accounts_payable, 2),
            "Short_Term_Debt": round(short_term_debt, 2),
            "Current_Liabilities": round(current_liabilities, 2),
            "Long_Term_Debt": round(long_term_debt, 2),
            "Total_Liabilities": round(total_liabilities, 2),
            "Equity": round(equity, 2)
        })
    
    df = pd.DataFrame(statements)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_periods} financial statement periods: {output_path}")
    return df


def generate_sales_orders(num_orders=500, output_path="data/synthetic_sales_orders.csv"):
    """Generate synthetic Sales Order data."""
    
    np.random.seed(42)
    base_date = datetime(2024, 1, 1)
    
    products = [
        "Product A", "Product B", "Product C", "Product D", "Product E",
        "Product F", "Product G", "Product H", "Product I", "Product J"
    ]
    
    customers = [f"CUST{str(i).zfill(5)}" for i in range(1, 101)]
    regions = ["North", "South", "East", "West", "Central"]
    
    orders = []
    
    for i in range(num_orders):
        order_date = base_date + timedelta(days=np.random.randint(0, 365))
        delivery_date = order_date + timedelta(days=np.random.randint(7, 30))
        
        customer = np.random.choice(customers)
        product = np.random.choice(products)
        quantity = np.random.randint(1, 100)
        unit_price = np.random.uniform(10, 500)
        total_amount = quantity * unit_price
        region = np.random.choice(regions)
        
        order_status = np.random.choice(
            ["Open", "In Process", "Delivered", "Cancelled"],
            p=[0.2, 0.3, 0.4, 0.1]
        )
        
        orders.append({
            "Order_Number": f"SO{str(i+1).zfill(6)}",
            "Order_Date": order_date.strftime("%Y-%m-%d"),
            "Delivery_Date": delivery_date.strftime("%Y-%m-%d"),
            "Customer_ID": customer,
            "Customer_Name": f"Customer {customer}",
            "Product_Code": f"PRD{str(np.random.randint(1, 100)).zfill(3)}",
            "Product_Name": product,
            "Quantity": quantity,
            "Unit_Price": round(unit_price, 2),
            "Total_Amount": round(total_amount, 2),
            "Currency": "USD",
            "Region": region,
            "Status": order_status,
            "Sales_Rep": f"REP{str(np.random.randint(1, 20)).zfill(2)}"
        })
    
    df = pd.DataFrame(orders)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_orders} sales orders: {output_path}")
    return df


def generate_all_datasets():
    """Generate all synthetic datasets."""
    print("Generating synthetic SAP finance datasets...")
    generate_gl_accounts()
    generate_financial_statements()
    generate_sales_orders()
    print("All datasets generated successfully!")


if __name__ == "__main__":
    generate_all_datasets()


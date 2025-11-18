"""
SAP OData Connector

Implements SAPFinanceConnector class for connecting to SAP OData services
and fetching sales orders, products, line items, and business partners.
"""

import os
import json
import logging
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SAPFinanceConnector:
    """Connector for SAP OData API services."""
    
    def __init__(self, verify_ssl=False):
        """
        Initialize the SAP Finance Connector.
        
        Args:
            verify_ssl: Whether to verify SSL certificates
        """
        self.user = os.getenv("SAP_USERNAME")
        self.pw = os.getenv("SAP_PASSWORD")
        self.base = os.getenv("SAP_BASE_URL", "https://sapes5.sapdevcenter.com/sap/opu/odata/IWBEP/GWSAMPLE_BASIC")
        self.client = os.getenv("SAP_CLIENT", "002")
        self.headers = {"Accept": "application/json", "x-csrf-token": "Fetch"}
        self.cookies = None
        self.verify_ssl = verify_ssl
        
        if not self.user or not self.pw:
            logging.warning("SAP_USERNAME or SAP_PASSWORD environment variable not set.")

    def test_connection(self):
        """
        Test the connection to SAP OData service.
        
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.user or not self.pw:
            return False, "SAP credentials not set in environment variables."
        
        metadata_url = f"{self.base}/$metadata"
        try:
            logging.info(f"Attempting to connect to SAP metadata URL: {metadata_url} with client {self.client}")
            r = requests.get(
                metadata_url,
                auth=(self.user, self.pw),
                headers={"Accept": "application/xml"},
                params={"sap-client": self.client},
                verify=self.verify_ssl,
                timeout=20
            )
            r.raise_for_status()
            self.cookies = r.cookies
            tok = r.headers.get("x-csrf-token")
            if tok:
                self.headers['x-csrf-token'] = tok
                logging.info("SAP Connection successful, CSRF token fetched.")
                return True, "Connected successfully."
            else:
                logging.warning("SAP Connection successful, but x-csrf-token not found.")
                return True, "Connected (Warning: CSRF token missing)."
        except requests.exceptions.Timeout:
            logging.error(f"SAP connection timed out: {metadata_url}")
            return False, "Connection timed out."
        except requests.exceptions.HTTPError as e:
            logging.error(f"SAP connection HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            return False, f"Connection failed (HTTP {e.response.status_code}). Check URL/Credentials/Client."
        except requests.exceptions.RequestException as e:
            logging.error(f"SAP connection failed: {e}")
            return False, f"Connection failed: {type(e).__name__}. Check network/URL."
        except Exception as e:
            logging.error(f"An unexpected error occurred during SAP connection test: {e}", exc_info=True)
            return False, f"An unexpected error occurred: {e}"

    def fetch(self, entity, top):
        """
        Fetch data from a specific OData entity.
        
        Args:
            entity: Entity name (e.g., "SalesOrderSet")
            top: Maximum number of records to fetch
            
        Returns:
            List of records as dictionaries
        """
        if not self.cookies or 'x-csrf-token' not in self.headers.get('x-csrf-token', ''):
            logging.warning(f"Attempting to fetch {entity} without established connection/CSRF token.")
            connected, msg = self.test_connection()
            if not connected:
                logging.error(f"Cannot fetch {entity}, SAP connection failed: {msg}")
                raise ConnectionError(f"SAP Connection failed: {msg}")
            elif 'x-csrf-token' not in self.headers.get('x-csrf-token', ''):
                logging.warning(f"Proceeding to fetch {entity} without CSRF token. May fail.")

        url = f"{self.base}/{entity}"
        params = {
            "sap-client": self.client,
            "$format": "json",
            "$top": str(top)
        }
        logging.info(f"Fetching data from: {url} with params: {params}")
        try:
            r = requests.get(url, params=params, auth=(self.user, self.pw), headers=self.headers,
                             cookies=self.cookies, verify=self.verify_ssl, timeout=30)
            r.raise_for_status()
            content_type = r.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                # Handle potential empty response or structure variations
                response_json = r.json()
                data = response_json.get('d', {}).get('results', []) if isinstance(response_json.get('d'), dict) else []
                logging.info(f"Successfully fetched {len(data)} records from {entity}.")
                return data
            else:
                logging.error(f"Unexpected Content-Type '{content_type}' for {entity}. Response: {r.text[:200]}")
                raise ValueError(f"Expected JSON response, got {content_type}")

        except requests.exceptions.Timeout:
            logging.error(f"Timeout occurred while fetching {entity} from {url}")
            raise TimeoutError(f"Timeout fetching {entity}")
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error fetching {entity}: {e.response.status_code} - {e.response.text[:200]}")
            raise ConnectionError(f"HTTP {e.response.status_code} fetching {entity}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {entity}: {e}")
            raise ConnectionError(f"Request failed for {entity}: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON for {entity}: {e}. Response: {r.text[:500]}")
            raise ValueError(f"Invalid JSON received for {entity}")
        except Exception as e:
            logging.error(f"Unexpected error fetching {entity}: {e}", exc_info=True)
            raise

    def fetch_orders(self, top=500):
        """
        Fetch sales orders.
        
        Args:
            top: Maximum number of records
            
        Returns:
            List of sales order records
        """
        return self.fetch("SalesOrderSet", top)
    
    def fetch_products(self, top=500):
        """
        Fetch products.
        
        Args:
            top: Maximum number of records
            
        Returns:
            List of product records
        """
        return self.fetch("ProductSet", top)
    
    def fetch_line_items(self, top=400):
        """
        Fetch sales order line items.
        
        Args:
            top: Maximum number of records
            
        Returns:
            List of line item records
        """
        return self.fetch("SalesOrderLineItemSet", top)
    
    def fetch_partners(self, top=500):
        """
        Fetch business partners.
        
        Args:
            top: Maximum number of records
            
        Returns:
            List of business partner records
        """
        return self.fetch("BusinessPartnerSet", top)
    
    def fetch_orders_df(self, top=500):
        """
        Fetch sales orders as a pandas DataFrame.
        
        Args:
            top: Maximum number of records
            
        Returns:
            pandas DataFrame
        """
        data = self.fetch_orders(top)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()
    
    def fetch_products_df(self, top=500):
        """
        Fetch products as a pandas DataFrame.
        
        Args:
            top: Maximum number of records
            
        Returns:
            pandas DataFrame
        """
        data = self.fetch_products(top)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()
    
    def fetch_line_items_df(self, top=400):
        """
        Fetch sales order line items as a pandas DataFrame.
        
        Args:
            top: Maximum number of records
            
        Returns:
            pandas DataFrame
        """
        data = self.fetch_line_items(top)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()
    
    def fetch_partners_df(self, top=500):
        """
        Fetch business partners as a pandas DataFrame.
        
        Args:
            top: Maximum number of records
            
        Returns:
            pandas DataFrame
        """
        data = self.fetch_partners(top)
        if data:
            return pd.DataFrame(data)
        return pd.DataFrame()


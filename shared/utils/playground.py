"""
SAP-RPT-1-OSS Playground Utilities

Functions for handling dataset uploads, previews, training, and results export.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import tempfile
import os


def load_dataset(file_path: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Load dataset from file (CSV, Parquet, or JSON).
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Tuple of (DataFrame, error_message)
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_ext in ['.json', '.jsonl']:
            df = pd.read_json(file_path, lines=(file_ext == '.jsonl'))
        else:
            return None, f"Unsupported file format: {file_ext}"
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def detect_task_type(filename: str) -> str:
    """
    Detect task type from filename.
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        'classification' or 'regression'
    """
    filename_lower = filename.lower()
    if 'classification' in filename_lower:
        return 'classification'
    elif 'regression' in filename_lower:
        return 'regression'
    return 'classification'  # Default


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'columns': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }
    return info


def auto_select_target_column(df: pd.DataFrame, task_type: str) -> Optional[str]:
    """
    Auto-select target column (defaults to last column).
    
    Args:
        df: DataFrame
        task_type: 'classification' or 'regression'
        
    Returns:
        Column name or None
    """
    if len(df.columns) == 0:
        return None
    
    # Default to last column
    target = df.columns[-1]
    
    # If task type is regression, prefer numeric columns
    if task_type == 'regression':
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            # Prefer last numeric column
            for col in reversed(numeric_cols):
                if col == target or df[col].dtype in [np.float64, np.int64]:
                    return col
    
    return target


def detect_task_type_from_column(df: pd.DataFrame, target_column: str) -> str:
    """
    Detect task type from target column's data type.
    
    Args:
        df: DataFrame
        target_column: Name of target column
        
    Returns:
        'classification' or 'regression'
    """
    if target_column not in df.columns:
        return 'classification'  # Default
    
    target_series = df[target_column]
    
    # Check if numeric
    if not pd.api.types.is_numeric_dtype(target_series):
        return 'classification'  # Non-numeric = classification
    
    # Check if integer-like
    try:
        unique_values = target_series.dropna().nunique()
        if unique_values <= 20:  # Few unique values = likely classification
            # Check if values are integer-like
            sample = target_series.dropna().head(100)
            int_values = sample.astype(int)
            float_values = sample.astype(float)
            if (int_values == float_values).all():
                return 'classification'
        
        # Many unique numeric values = regression
        return 'regression'
    except:
        return 'regression'  # Default to regression for numeric


def prepare_train_test_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Prepare train/test split.
    
    Args:
        df: Full dataset
        target_column: Name of target column
        test_size: Proportion of test set (0.1 to 0.5)
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test)
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values in target
    valid_mask = ~y.isnull()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # Calculate split index
    n_total = len(X)
    n_test = int(n_total * test_size)
    n_train = n_total - n_test
    
    if n_test == 0:
        n_test = 1
        n_train = n_total - 1
    
    # Split
    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_test = X.iloc[n_train:n_train + n_test]
    y_test = y.iloc[n_train:n_train + n_test]
    
    return X_train, y_train, X_test, y_test


def preprocess_data(
    df: pd.DataFrame,
    handle_missing: str = 'mean',
    normalize: bool = False
) -> pd.DataFrame:
    """
    Preprocess dataset.
    
    Args:
        df: DataFrame to preprocess
        handle_missing: How to handle missing values ('mean', 'median', 'drop', 'zero')
        normalize: Whether to normalize numeric columns
        
    Returns:
        Preprocessed DataFrame
    """
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    
    if handle_missing == 'mean':
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
            df_processed[numeric_cols].mean()
        )
    elif handle_missing == 'median':
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(
            df_processed[numeric_cols].median()
        )
    elif handle_missing == 'zero':
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(0)
    elif handle_missing == 'drop':
        df_processed = df_processed.dropna(subset=numeric_cols)
    
    # Normalize numeric columns
    if normalize:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed


def export_results(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: np.ndarray,
    task_type: str,
    filename_prefix: str = "results"
) -> str:
    """
    Export results to CSV file.
    
    Args:
        X_test: Test features
        y_test: True target values
        predictions: Model predictions
        task_type: 'classification' or 'regression'
        filename_prefix: Prefix for output filename
        
    Returns:
        Path to exported CSV file
    """
    # Create results DataFrame
    results_df = X_test.copy()
    results_df['true_value'] = y_test.values
    
    if task_type == 'classification':
        results_df['predicted_class'] = predictions
    else:
        # Regression - format with comma for decimal separator
        results_df['predicted_value'] = predictions
    
    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"{filename_prefix}_results.csv")
    
    results_df.to_csv(output_path, index=False)
    
    return output_path


def check_embedding_server() -> Tuple[bool, str]:
    """
    Check if embedding server is running.
    
    Returns:
        Tuple of (is_running, message)
    """
    # First check if packages are available (but don't fail if import path is different)
    try:
        import zmq
    except ImportError:
        return False, "pyzmq package not installed. Install with: pip install pyzmq"
    
    # Try to check if server is running (don't check package installation here)
    try:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.settimeout(1000)  # 1 second timeout
        socket.connect("tcp://localhost:5555")
        socket.send_string("ping")
        
        # Try to receive with timeout
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        if poller.poll(1000):  # 1 second timeout
            response = socket.recv_string()
            socket.close()
            context.term()
            return True, "Embedding server is running and responding"
        else:
            socket.close()
            context.term()
            return False, "Embedding server not responding on port 5555"
    except zmq.ZMQError as e:
        if "Connection refused" in str(e) or "No such file or directory" in str(e):
            return False, "Embedding server is not running"
        return False, f"Connection error: {str(e)}"
    except Exception as e:
        return False, f"Error checking embedding server: {str(e)}"


def is_sap_rpt_oss_installed() -> bool:
    """Check if sap-rpt-oss package is installed."""
    try:
        import sap_rpt_oss
        return True
    except ImportError:
        try:
            # Try alternative import paths
            from sap_rpt_oss import SAP_RPT_OSS_Classifier
            return True
        except ImportError:
            return False


def start_embedding_server(gpu_idx: Optional[int] = None) -> Tuple[bool, str]:
    """
    Start the embedding server automatically.
    
    Args:
        gpu_idx: GPU index to use (None for CPU)
        
    Returns:
        Tuple of (success, message)
    """
    # Check if package is installed
    if not is_sap_rpt_oss_installed():
        return False, "sap-rpt-oss package not found. Install with: pip install git+https://github.com/SAP-samples/sap-rpt-1-oss"
    
    # Check if server is already running
    is_running, _ = check_embedding_server()
    if is_running:
        return True, "Embedding server is already running"
    
    try:
        # Try multiple import paths
        start_func = None
        import_paths = [
            "sap_rpt_oss.scripts.start_embedding_server",
            "sap_rpt_oss.start_embedding_server",
            "sap_rpt_oss.data.tokenizer",  # Sometimes server is in tokenizer
        ]
        
        for import_path in import_paths:
            try:
                module = __import__(import_path, fromlist=['start_embedding_server'])
                if hasattr(module, 'start_embedding_server'):
                    start_func = getattr(module, 'start_embedding_server')
                    break
            except (ImportError, AttributeError):
                continue
        
        if start_func is None:
            # Try using threading approach - the server might start automatically when model is used
            # For now, just inform user that server will start when needed
            return False, "Embedding server will start automatically when the model makes predictions. No manual start needed."
        
        # Use threading to start server in background (simpler than subprocess)
        import threading
        
        def run_server():
            try:
                start_func(
                    sentence_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                    gpu_idx=gpu_idx
                )
            except Exception as e:
                # Server might block, that's okay
                pass
        
        # Start server in background thread
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        import time
        time.sleep(3)
        
        # Check if it started
        is_running, message = check_embedding_server()
        if is_running:
            return True, "Embedding server started successfully in background"
        else:
            # Server might take longer to start, or it starts on-demand
            return False, "Server thread started. The embedding server will be available when the model needs it (starts on-demand during predictions)."
            
    except Exception as e:
        # If we can't start it manually, that's okay - SAP-RPT-OSS might start it automatically
        return False, f"Manual start not available: {str(e)}. The embedding server will start automatically when the model makes predictions."


def ensure_embedding_server_running() -> Tuple[bool, str]:
    """
    Ensure embedding server is running, start it if not.
    Note: SAP-RPT-OSS may start the server automatically when needed.
    
    Returns:
        Tuple of (is_running, message)
    """
    is_running, message = check_embedding_server()
    if is_running:
        return True, message
    
    # Try to start it (but don't fail if we can't - server may start on-demand)
    success, start_message = start_embedding_server(None)
    if success:
        return True, f"Auto-started: {start_message}"
    else:
        # Server not running, but SAP-RPT-OSS may start it automatically when model is used
        # This is not a fatal error - the model will attempt to start it when needed
        return False, f"Server not currently running. {start_message}"


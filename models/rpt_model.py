"""
SAP-RPT-1-OSS Model Wrapper

Provides a wrapper for SAP-RPT-OSS-Classifier and Regressor with
authentication handling and CPU fallback options.
"""

import os
import logging
from typing import Optional, Union
import pandas as pd
import numpy as np
from huggingface_hub import login as hf_login
from dotenv import load_dotenv

# Try to import SAP-RPT-OSS models
try:
    from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
    SAP_RPT_AVAILABLE = True
except ImportError:
    SAP_RPT_AVAILABLE = False
    logging.warning("sap-rpt-oss package not installed. Install with: pip install git+https://github.com/SAP-samples/sap-rpt-1-oss")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RPTModelWrapper:
    """Wrapper for SAP-RPT-1-OSS models with authentication and resource management."""
    
    def __init__(self, model_type: str = "classifier", max_context_size: int = 2048, bagging: int = 1):
        """
        Initialize the RPT model wrapper.
        
        Args:
            model_type: "classifier" or "regressor"
            max_context_size: Maximum context size (8192 for best performance, 2048 for CPU)
            bagging: Bagging factor (8 for best performance, 1 for lightweight)
        """
        if not SAP_RPT_AVAILABLE:
            raise ImportError("sap-rpt-oss package is not installed. Please install it first.")
        
        self.model_type = model_type.lower()
        self.max_context_size = max_context_size
        self.bagging = bagging
        self.model = None
        self.is_fitted = False
        
        # Check for Hugging Face token
        self._check_hf_authentication()
        
        # Initialize model
        self._initialize_model()
    
    def _check_hf_authentication(self):
        """Check and handle Hugging Face authentication."""
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if hf_token:
            try:
                hf_login(token=hf_token)
                logger.info("Hugging Face authentication successful using token from environment.")
            except Exception as e:
                logger.warning(f"Failed to login with token: {e}. Trying interactive login...")
                try:
                    hf_login()
                except Exception as e2:
                    logger.error(f"Hugging Face authentication failed: {e2}")
        else:
            logger.warning("HUGGINGFACE_TOKEN not found in environment. Attempting interactive login...")
            try:
                hf_login()
            except Exception as e:
                logger.error(f"Hugging Face authentication failed: {e}")
                logger.info("Please set HUGGINGFACE_TOKEN in .env file or run: huggingface-cli login")
    
    def _initialize_model(self):
        """Initialize the appropriate model based on type."""
        try:
            if self.model_type == "classifier":
                self.model = SAP_RPT_OSS_Classifier(
                    max_context_size=self.max_context_size,
                    bagging=self.bagging
                )
                logger.info(f"Initialized SAP-RPT-OSS-Classifier with context_size={self.max_context_size}, bagging={self.bagging}")
            elif self.model_type == "regressor":
                self.model = SAP_RPT_OSS_Regressor(
                    max_context_size=self.max_context_size,
                    bagging=self.bagging
                )
                logger.info(f"Initialized SAP-RPT-OSS-Regressor with context_size={self.max_context_size}, bagging={self.bagging}")
            else:
                raise ValueError(f"Invalid model_type: {self.model_type}. Must be 'classifier' or 'regressor'")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fit the model on training data.
        
        Args:
            X: Feature data (DataFrame or array)
            y: Target data (Series or array)
        """
        try:
            if isinstance(X, np.ndarray):
                # Convert to DataFrame if needed
                X = pd.DataFrame(X)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            
            logger.info(f"Fitting model on {len(X)} samples...")
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info("Model fitting completed successfully.")
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Make predictions.
        
        Args:
            X: Feature data (DataFrame or array)
            
        Returns:
            Predictions (array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            
            logger.info(f"Making predictions on {len(X)} samples...")
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Predict class probabilities (classification only).
        
        Args:
            X: Feature data (DataFrame or array)
            
        Returns:
            Probability predictions (array)
        """
        if self.model_type != "classifier":
            raise ValueError("predict_proba() is only available for classifiers.")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        try:
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            
            logger.info(f"Predicting probabilities on {len(X)} samples...")
            probabilities = self.model.predict_proba(X)
            return probabilities
        except Exception as e:
            logger.error(f"Error during probability prediction: {e}")
            raise
    
    def get_model_info(self):
        """Get information about the current model configuration."""
        return {
            "model_type": self.model_type,
            "max_context_size": self.max_context_size,
            "bagging": self.bagging,
            "is_fitted": self.is_fitted,
            "sap_rpt_available": SAP_RPT_AVAILABLE
        }


def create_model(model_type: str = "classifier", use_gpu: bool = True):
    """
    Factory function to create a model with appropriate settings.
    
    Args:
        model_type: "classifier" or "regressor"
        use_gpu: Whether to use GPU-optimized settings (requires 80GB GPU memory)
    
    Returns:
        RPTModelWrapper instance
    """
    if use_gpu:
        # Best performance settings (requires 80GB GPU)
        return RPTModelWrapper(
            model_type=model_type,
            max_context_size=8192,
            bagging=8
        )
    else:
        # CPU-friendly settings
        return RPTModelWrapper(
            model_type=model_type,
            max_context_size=2048,
            bagging=1
        )


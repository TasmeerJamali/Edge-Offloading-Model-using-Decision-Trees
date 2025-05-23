import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        """Initialize the DataPreprocessor with necessary encoders and scalers."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'cpu_usage', 'battery_level', 'network_latency',
            'task_complexity', 'bandwidth', 'data_size'
        ]
        
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Preprocess the dataset for model training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
                - X: Preprocessed features
                - y: Target variable
                - preprocessor_info: Dictionary containing preprocessing information
        """
        logger.info("Starting data preprocessing")
        
        # Encode categorical variables
        df['task_complexity'] = self.label_encoder.fit_transform(df['task_complexity'])
        
        # Scale numerical features
        X = df[self.feature_columns].values
        X = self.scaler.fit_transform(X)
        
        # Get target variable
        y = df['target'].values
        
        # Store preprocessing information
        preprocessor_info = {
            'feature_columns': self.feature_columns,
            'label_encoder_classes': self.label_encoder.classes_,
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_
        }
        
        logger.info("Data preprocessing completed")
        return X, y, preprocessor_info
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                  test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target variable
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split completed. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_single_sample(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess a single sample for prediction.
        
        Args:
            sample (Dict[str, Any]): Dictionary containing feature values
            
        Returns:
            np.ndarray: Preprocessed features
        """
        # Convert task complexity to numerical
        sample['task_complexity'] = self.label_encoder.transform([sample['task_complexity']])[0]
        
        # Create feature array
        features = np.array([[sample[col] for col in self.feature_columns]])
        
        # Scale features
        features = self.scaler.transform(features)
        
        return features

    def load_fitted(self, encoder_path, scaler_path):
        self.label_encoder = joblib.load(encoder_path)
        self.scaler = joblib.load(scaler_path) 
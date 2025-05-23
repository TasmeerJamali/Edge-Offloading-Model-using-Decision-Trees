import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, num_samples: int = 10000):
        """
        Initialize the DataGenerator with specified number of samples.
        
        Args:
            num_samples (int): Number of samples to generate
        """
        self.num_samples = num_samples
        logger.info(f"Initialized DataGenerator with {num_samples} samples")
        
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate synthetic dataset for edge computing offloading decisions.
        
        Returns:
            pd.DataFrame: Generated dataset with features and target
        """
        data = {
            'user_id': np.random.randint(1, 101, self.num_samples),  # 100 different users
            'cpu_usage': np.random.uniform(0, 100, self.num_samples),
            'battery_level': np.random.uniform(0, 100, self.num_samples),
            'network_latency': np.random.uniform(10, 1000, self.num_samples),  # ms
            'task_complexity': np.random.choice(['low', 'medium', 'high'], self.num_samples),
            'bandwidth': np.random.uniform(100, 10000, self.num_samples),  # kbps
            'data_size': np.random.uniform(1, 1000, self.num_samples)  # KB
        }
        
        df = pd.DataFrame(data)
        df['target'] = self._apply_decision_rules(df)
        logger.info(f"Generated dataset with shape: {df.shape}")
        return df
    
    def _apply_decision_rules(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply mathematical decision rules to determine offloading decisions.
        
        Args:
            df (pd.DataFrame): Input dataframe with features
            
        Returns:
            np.ndarray: Binary decisions (0 for local, 1 for offload)
        """
        # Energy consumption calculations
        complexity_map = {'low': 1, 'medium': 2, 'high': 3}
        E_loc = df['cpu_usage'] * df['task_complexity'].map(complexity_map)
        E_off = df['network_latency'] * df['data_size'] / df['bandwidth']
        
        # Time calculations
        T_loc = E_loc * 1.5  # Local execution time
        T_net = df['data_size'] / df['bandwidth'] * 1000  # Network transmission time
        T_ES = df['task_complexity'].map({'low': 100, 'medium': 200, 'high': 300})
        T_cloud = T_ES * 1.2  # Cloud processing time
        
        # Cost calculations
        lambda_weight = 0.5
        J_loc = lambda_weight * E_loc + (1 - lambda_weight) * T_loc
        J_off = lambda_weight * E_off + (1 - lambda_weight) * (T_net + T_ES + T_cloud)
        
        # Apply constraints
        battery_threshold = 20  # Minimum battery level for local execution
        bandwidth_threshold = 500  # Minimum bandwidth for offloading
        
        # Final decision considering constraints
        decision = (
            (J_loc > J_off) & 
            (df['battery_level'] > battery_threshold) & 
            (df['bandwidth'] > bandwidth_threshold)
        ).astype(int)
        
        return decision

    def save_dataset(self, df: pd.DataFrame, path: str) -> None:
        """
        Save the generated dataset to a CSV file.
        
        Args:
            df (pd.DataFrame): Dataset to save
            path (str): Path to save the dataset
        """
        df.to_csv(path, index=False)
        logger.info(f"Dataset saved to {path}") 
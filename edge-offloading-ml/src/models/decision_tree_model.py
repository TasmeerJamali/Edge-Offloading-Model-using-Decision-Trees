import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import joblib
import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EdgeOffloadingModel:
    def __init__(self):
        """Initialize the Edge Offloading Decision Tree model."""
        self.model = DecisionTreeClassifier(
            max_depth=5,  # Control complexity
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        logger.info("Initialized Decision Tree model")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the decision tree model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        self.model.fit(X_train, y_train)
        logger.info("Model training completed")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Predictions (0 for local, 1 for offload)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict[str, float]: Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {metrics['accuracy']:.2f}")
        return metrics
    
    def visualize_tree(self, feature_names: list, class_names: list) -> None:
        """
        Visualize the decision tree structure.
        
        Args:
            feature_names (list): Names of the features
            class_names (list): Names of the classes
        """
        plt.figure(figsize=(20,10))
        tree.plot_tree(self.model, 
                      feature_names=feature_names,
                      class_names=class_names,
                      filled=True,
                      rounded=True)
        plt.savefig('models/visualizations/decision_tree.png')
        plt.close()
        logger.info("Decision tree visualization saved")
    
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_test (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/visualizations/confusion_matrix.png')
        plt.close()
        logger.info("Confusion matrix visualization saved")
    
    def plot_feature_importance(self, feature_names: list) -> None:
        """
        Plot and save feature importance.
        
        Args:
            feature_names (list): Names of the features
        """
        importance = self.model.feature_importances_
        plt.figure(figsize=(10,6))
        plt.bar(feature_names, importance)
        plt.xticks(rotation=45)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('models/visualizations/feature_importance.png')
        plt.close()
        logger.info("Feature importance visualization saved")
    
    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path (str): Path to save the model
        """
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path (str): Path to load the model from
        """
        self.model = joblib.load(path)
        logger.info(f"Model loaded from {path}") 
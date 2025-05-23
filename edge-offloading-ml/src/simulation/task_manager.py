import numpy as np
import joblib
import logging
from typing import Dict, Any, Tuple
from ..preprocessing.data_preprocessor import DataPreprocessor
from .rule_based_offloader import RuleBasedOffloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, model_path: str):
        """
        Initialize the TaskManager with a trained model.
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = joblib.load(model_path)
        self.preprocessor = DataPreprocessor()
        self.preprocessor.load_fitted('models/trained/label_encoder.joblib', 'models/trained/scaler.joblib')
        self.rule_based = RuleBasedOffloader()
        logger.info("TaskManager initialized with model from {model_path}")
        
    def make_offload_decision(self, task_params: Dict[str, Any]) -> Tuple[int, float]:
        """
        Make offloading decision based on task parameters.
        
        Args:
            task_params (Dict[str, Any]): Dictionary containing task parameters
            
        Returns:
            Tuple[int, float]: Decision (0 for local, 1 for offload) and confidence
        """
        # Preprocess the input
        features = self.preprocessor.preprocess_single_sample(task_params)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        confidence = probability[prediction]
        
        logger.info(f"Decision made: {'Offload' if prediction == 1 else 'Local'} "
                   f"with confidence: {confidence:.2f}")
        
        return prediction, confidence
    
    def get_decision_explanation(self, task_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get explanation for the offloading decision.
        
        Args:
            task_params (Dict[str, Any]): Dictionary containing task parameters
            
        Returns:
            Dict[str, Any]: Dictionary containing decision explanation
        """
        # Get ML model decision
        ml_prediction, ml_confidence = self.make_offload_decision(task_params)
        
        # Map integer task_complexity back to string for rule-based system
        complexity_map = {0: 'low', 1: 'medium', 2: 'high'}
        task_params_rule = task_params.copy()
        if isinstance(task_params_rule['task_complexity'], (int, np.integer)):
            task_params_rule['task_complexity'] = complexity_map[int(task_params_rule['task_complexity'])]
        
        # Get rule-based decision
        rule_prediction, rule_explanation = self.rule_based.make_decision(task_params_rule)
        
        # Prepare comprehensive explanation
        explanation = {
            'ml_decision': 'Offload to Cloud' if ml_prediction == 1 else 'Execute Locally',
            'ml_confidence': ml_confidence,
            'rule_based_decision': rule_explanation['decision'],
            'comparison': {
                'ml_factors': {
                    'cpu_usage': task_params['cpu_usage'],
                    'battery_level': task_params['battery_level'],
                    'network_latency': task_params['network_latency'],
                    'task_complexity': task_params['task_complexity'],
                    'bandwidth': task_params['bandwidth'],
                    'data_size': task_params['data_size']
                },
                'rule_based_factors': rule_explanation['factors']
            },
            'agreement': ml_prediction == rule_prediction
        }
        
        return explanation 
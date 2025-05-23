import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.decision_tree_model import EdgeOffloadingModel
from src.data_generation.synthetic_data_generator import DataGenerator
import logging
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create necessary directories
    os.makedirs('models/visualizations', exist_ok=True)
    os.makedirs('models/trained', exist_ok=True)
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    data_generator = DataGenerator(num_samples=10000)
    df = data_generator.generate_dataset()
    
    # Save the dataset
    df.to_csv('data/processed/edge_offloading_dataset.csv', index=False)
    logger.info("Dataset saved to data/processed/edge_offloading_dataset.csv")
    
    # Prepare features and target
    feature_columns = ['cpu_usage', 'battery_level', 'network_latency',
                      'task_complexity', 'bandwidth', 'data_size']
    le = LabelEncoder()
    df['task_complexity'] = le.fit_transform(df['task_complexity'])
    X = df[feature_columns]
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    logger.info("Training Decision Tree model...")
    model = EdgeOffloadingModel()
    model.train(X_train, y_train)
    
    # Evaluate the model
    logger.info("Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    feature_names = X.columns.tolist()
    class_names = ['Local', 'Offload']
    
    model.visualize_tree(feature_names, class_names)
    model.plot_confusion_matrix(y_test, model.predict(X_test))
    model.plot_feature_importance(feature_names)
    
    # Save the trained model
    model.save_model('models/trained/edge_offloading_model.joblib')
    logger.info("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 
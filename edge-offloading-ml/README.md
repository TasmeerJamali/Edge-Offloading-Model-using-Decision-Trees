# Edge Computing Offloading Decision System

This project implements a Machine Learning-based decision system for edge computing task offloading. The system uses a Decision Tree Classifier to determine whether a task should be executed locally or offloaded to the cloud based on various system parameters.

## Features

- Decision Tree-based offloading decision making
- Real-time system parameter monitoring
- Visualization of decision tree and model performance
- Support for multiple users
- Comprehensive model evaluation metrics

## Project Structure

```
edge-offloading-ml/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── trained/
│   └── visualizations/
├── src/
│   ├── data_generation/
│   ├── models/
│   ├── preprocessing/
│   └── simulation/
├── main.py
├── train_model.py
└── requirements.txt
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train_model.py
```

2. Run the simulation interface:
```bash
streamlit run main.py
```

## Model Details

### Input Features
- CPU Usage (%)
- Battery Level (%)
- Network Latency (ms)
- Task Complexity (low, medium, high)
- Bandwidth Availability (kbps)
- Data Size (KB)
- User Information

### Output
Binary classification:
- 0: Execute Locally
- 1: Offload to Cloud

### Model Performance
The Decision Tree model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Visualization

The project generates several visualizations:
1. Decision Tree Structure
2. Confusion Matrix
3. Feature Importance
4. System Status Dashboard

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
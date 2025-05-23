import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from src.simulation.task_manager import TaskManager
from src.data_generation.synthetic_data_generator import DataGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Edge Computing Offloading Decision System",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if 'task_manager' not in st.session_state:
    st.session_state.task_manager = TaskManager('models/trained/edge_offloading_model.joblib')

def main():
    st.title("⚡ Edge Computing Offloading Decision System")
    
    # Sidebar for task parameters
    st.sidebar.header("Task Parameters")
    
    task_params = {
        'user_id': st.sidebar.number_input("User ID", 1, 100, 1),
        'cpu_usage': st.sidebar.slider("CPU Usage (%)", 0, 100, 50),
        'battery_level': st.sidebar.slider("Battery Level (%)", 0, 100, 80),
        'network_latency': st.sidebar.slider("Network Latency (ms)", 10, 1000, 100),
        'task_complexity': st.sidebar.selectbox("Task Complexity", ['low', 'medium', 'high']),
        'bandwidth': st.sidebar.slider("Bandwidth (kbps)", 100, 10000, 1000),
        'data_size': st.sidebar.slider("Data Size (KB)", 1, 1000, 100)
    }
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Decision Making")
        if st.button("Make Decision"):
            with st.spinner("Analyzing task parameters..."):
                explanation = st.session_state.task_manager.get_decision_explanation(task_params)
                
                # Display ML decision
                st.subheader("ML Model Decision")
                if explanation['ml_decision'] == 'Offload to Cloud':
                    st.success("🔄 Offload to Cloud")
                else:
                    st.success("💻 Execute Locally")
                st.metric("Confidence", f"{explanation['ml_confidence']:.2%}")
                
                # Display rule-based decision
                st.subheader("Rule-Based Decision")
                if explanation['rule_based_decision'] == 'Offload to Cloud':
                    st.info("🔄 Offload to Cloud")
                else:
                    st.info("💻 Execute Locally")
                
                # Display agreement
                st.subheader("Decision Agreement")
                if explanation['agreement']:
                    st.success("✅ ML and Rule-based decisions agree")
                else:
                    st.warning("⚠️ ML and Rule-based decisions differ")
                
                # Display factors
                st.subheader("Decision Factors")
                ml_factors = explanation['comparison']['ml_factors']
                rule_factors = explanation['comparison']['rule_based_factors']
                all_keys = list(set(ml_factors.keys()) | set(rule_factors.keys()))
                factors_df = pd.DataFrame({
                    'Factor': all_keys,
                    'ML Value': [ml_factors.get(k, '') for k in all_keys],
                    'Rule-Based Value': [rule_factors.get(k, '') for k in all_keys]
                })
                st.dataframe(factors_df)
    
    with col2:
        st.header("System Status")
        # Create visualizations
        fig = px.bar(
            x=list(task_params.keys()),
            y=list(task_params.values()),
            title="Current System Parameters"
        )
        st.plotly_chart(fig)
        
        # Display task parameters
        st.subheader("Task Parameters")
        st.json(task_params)

if __name__ == "__main__":
    main() 
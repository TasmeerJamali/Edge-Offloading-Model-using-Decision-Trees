import logging
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedOffloader:
    def __init__(self):
        """Initialize the Rule-Based Offloader with predefined thresholds."""
        # Thresholds for decision making
        self.BATTERY_THRESHOLD = 20  # Minimum battery level for local execution
        self.BANDWIDTH_THRESHOLD = 500  # Minimum bandwidth for offloading
        self.LATENCY_THRESHOLD = 100  # Maximum acceptable latency
        self.CPU_THRESHOLD = 80  # Maximum CPU usage for local execution
        self.DATA_SIZE_THRESHOLD = 500  # Maximum data size for local execution
        
        # Task complexity weights
        self.COMPLEXITY_WEIGHTS = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        
        logger.info("Initialized Rule-Based Offloader")
    
    def make_decision(self, task_params: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Make offloading decision based on predefined rules.
        
        Args:
            task_params (Dict[str, Any]): Dictionary containing task parameters
            
        Returns:
            Tuple[int, Dict[str, Any]]: Decision (0 for local, 1 for offload) and explanation
        """
        # Extract parameters
        battery = task_params['battery_level']
        bandwidth = task_params['bandwidth']
        latency = task_params['network_latency']
        cpu = task_params['cpu_usage']
        data_size = task_params['data_size']
        complexity = task_params['task_complexity']
        
        # Calculate local execution cost
        local_cost = (
            (cpu / 100) * self.COMPLEXITY_WEIGHTS[complexity] +
            (data_size / self.DATA_SIZE_THRESHOLD)
        )
        
        # Calculate offloading cost
        offload_cost = (
            (latency / self.LATENCY_THRESHOLD) +
            (data_size / bandwidth)
        )
        
        # Apply rules
        should_offload = (
            battery < self.BATTERY_THRESHOLD or  # Low battery
            (bandwidth > self.BANDWIDTH_THRESHOLD and  # Good bandwidth
             latency < self.LATENCY_THRESHOLD and  # Low latency
             offload_cost < local_cost)  # Lower cost to offload
        )
        
        # Prepare explanation
        explanation = {
            'decision': 'Offload to Cloud' if should_offload else 'Execute Locally',
            'factors': {
                'battery_level': f"{battery}% {'(Low)' if battery < self.BATTERY_THRESHOLD else '(OK)'}",
                'bandwidth': f"{bandwidth} kbps {'(Good)' if bandwidth > self.BANDWIDTH_THRESHOLD else '(Poor)'}",
                'latency': f"{latency} ms {'(Good)' if latency < self.LATENCY_THRESHOLD else '(Poor)'}",
                'cpu_usage': f"{cpu}% {'(High)' if cpu > self.CPU_THRESHOLD else '(OK)'}",
                'data_size': f"{data_size} KB {'(Large)' if data_size > self.DATA_SIZE_THRESHOLD else '(Small)'}",
                'task_complexity': complexity,
                'local_cost': f"{local_cost:.2f}",
                'offload_cost': f"{offload_cost:.2f}"
            }
        }
        
        logger.info(f"Rule-based decision: {'Offload' if should_offload else 'Local'}")
        return int(should_offload), explanation 
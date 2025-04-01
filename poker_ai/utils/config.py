from typing import Tuple, Dict, List, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for poker neural network"""
    input_dim: int
    num_actions: int
    hidden_layers: List[int] = None
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128]

@dataclass
class AgentConfig:
    """Configuration for reinforcement learning agent"""
    state_dim: int
    action_dim: int
    gamma: float = 0.95  # Discount factor
    epsilon_start: float = 1.0  # Initial exploration rate
    epsilon_end: float = 0.1  # Final exploration rate
    epsilon_decay: float = 0.995  # Exploration decay rate
    batch_size: int = 64  # Training batch size
    buffer_size: int = 10000  # Replay buffer size
    update_frequency: int = 4  # How often to train
    model_config: ModelConfig = None
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig(
                input_dim=self.state_dim,
                num_actions=self.action_dim
            )

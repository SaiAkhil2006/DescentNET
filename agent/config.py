"""
Hyperparameters and configuration settings for PPO training and evaluation.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class PPOConfig:
    """Hyperparameters for PPO Agent training on 6-DOF Lunar Lander."""
    # Policy Architecture
    policy_type: str = "MlpPolicy"
    net_arch: Dict[str, List[int]] = field(
        default_factory=lambda: dict(pi=[128, 128], vf=[128, 128])
    )
    activation_fn: str = "tanh"  # Standard for continuous control in robotics

    # Optimization
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005  # Slight entropy for attitude exploration
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training Horizon
    # 150,000 to 200,000 steps provides strong convergence within practical runtime
    total_timesteps: int = 200_000
    seed: int = 42

    # Vectorized Environments
    n_envs: int = 4

    # Logging and Checkpointing
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    log_interval: int = 5


# Directory Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BASELINE_DIR = os.path.join(MODELS_DIR, "baseline")
RANDOMIZED_DIR = os.path.join(MODELS_DIR, "randomized")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(RANDOMIZED_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

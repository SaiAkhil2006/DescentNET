"""
Training script for LunarPilot 6-DOF PPO Agent.
Supports training both the Baseline and Randomized Initial Conditions agents.
"""

import os
import argparse
import time
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from environment.lunar_env import LunarLander6DOFEnv
from environment.randomized_env import RandomizedLunarLander6DOFEnv
from agent.config import (
    PPOConfig,
    BASELINE_DIR,
    RANDOMIZED_DIR,
    MODELS_DIR,
)


def make_env_fn(env_class, seed: int, rank: int):
    """
    Utility factory to create seeded environments for vectorized execution.
    """
    def _init():
        env = env_class()
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    mode: str = "baseline",
    total_timesteps: int = 150_000,
    seed: int = 42,
    n_envs: int = 4,
    save_dir: str = None,
):
    """
    Train PPO agent.
    :param mode: 'baseline' or 'randomized'.
    :param total_timesteps: Total environment interaction steps across all parallel envs.
    :param seed: Random seed for reproducibility.
    :param n_envs: Number of vectorized parallel environments.
    :param save_dir: Directory to save policy weights and VecNormalize stats.
    """
    print("=" * 70)
    print(f"🚀 Starting LunarPilot 6-DOF Training — Mode: {mode.upper()}")
    print(f"   Timesteps: {total_timesteps:,} | Seed: {seed} | Vectorized Envs: {n_envs}")
    print("=" * 70)

    cfg = PPOConfig()
    set_random_seed(seed)

    if save_dir is None:
        save_dir = BASELINE_DIR if mode == "baseline" else RANDOMIZED_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Select environment class based on mode
    env_class = LunarLander6DOFEnv if mode == "baseline" else RandomizedLunarLander6DOFEnv

    # Vectorized Environment with Monitor wrapper
    env_fns = [make_env_fn(env_class, seed, i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    
    # VecNormalize normalizes observation dimensions to mean 0, variance 1
    # We do NOT normalize rewards to keep task reward scale physically interpretable
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )

    # Evaluation Environment
    eval_env = DummyVecEnv([make_env_fn(env_class, seed + 100, 0)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0, training=False)

    # Neural network policy kwargs
    policy_kwargs = dict(
        net_arch=cfg.net_arch,
        activation_fn=torch.nn.Tanh,
    )

    # Instantiate PPO Agent
    model = PPO(
        policy=cfg.policy_type,
        env=vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=save_dir,
        eval_freq=max(1000, cfg.eval_freq // n_envs),
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000, 25_000 // n_envs),
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix=f"ppo_lunar_{mode}",
    )

    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=False,
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user. Saving current model checkpoint...")

    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Training completed in {elapsed_time:.1f} seconds ({elapsed_time/60.0:.2f} minutes).")

    # Save final model and VecNormalize statistics
    final_model_path = os.path.join(save_dir, f"ppo_lunar_{mode}_final.zip")
    vec_norm_path = os.path.join(save_dir, f"vec_normalize_{mode}.pkl")

    model.save(final_model_path)
    vec_env.save(vec_norm_path)

    print(f"💾 Model saved to: {final_model_path}")
    print(f"💾 VecNormalize stats saved to: {vec_norm_path}")

    vec_env.close()
    eval_env.close()
    return final_model_path, vec_norm_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 6-DOF Lunar Lander PPO Agent")
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "randomized"],
        help="Training mode: 'baseline' (nominal initial distribution) or 'randomized' (broadened initial distribution)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=150_000,
        help="Total training timesteps across all environments (default: 150,000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of vectorized parallel environments (default: 4)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Custom directory to store checkpoints and models",
    )

    args = parser.parse_args()
    train(
        mode=args.mode,
        total_timesteps=args.timesteps,
        seed=args.seed,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
    )

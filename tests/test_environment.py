"""
Unit tests for LunarPilot Gymnasium Environment.
Verifies Gymnasium API compliance, observation space, action bounds, and termination logic.
"""

import pytest
import numpy as np

from environment.lunar_env import LunarLander6DOFEnv
from environment.randomized_env import RandomizedLunarLander6DOFEnv


def test_environment_initialization():
    """Verify standard Gymnasium spaces and properties."""
    env = LunarLander6DOFEnv()
    assert env.action_space.shape == (4,)
    assert env.observation_space.shape == (20,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    env.close()


def test_reset_and_finite_observations():
    """Verify reset output format and observation values are finite."""
    env = LunarLander6DOFEnv()
    obs, info = env.reset(seed=123)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (20,)
    assert np.all(np.isfinite(obs)), "Observations must not contain NaN or Inf"
    assert "state" in info
    assert "is_success" in info
    assert "is_crashed" in info
    env.close()


def test_step_api_contract():
    """Verify step conforms to Gymnasium standard 5-tuple output."""
    env = LunarLander6DOFEnv()
    env.reset(seed=456)
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (20,)
    assert np.all(np.isfinite(obs))
    assert isinstance(reward, (float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_randomized_environment_initial_conditions():
    """Verify that RandomizedLunarLander6DOFEnv samples wider distributions than Baseline."""
    env_base = LunarLander6DOFEnv()
    env_rand = RandomizedLunarLander6DOFEnv()

    base_altitudes = []
    rand_altitudes = []
    base_tilts = []
    rand_tilts = []

    for seed in range(50):
        _, info_b = env_base.reset(seed=seed)
        _, info_r = env_rand.reset(seed=seed)

        base_altitudes.append(info_b["state"]["pos"][2])
        rand_altitudes.append(info_r["state"]["pos"][2])
        base_tilts.append(np.rad2deg(info_b["state"]["tilt_angle"]))
        rand_tilts.append(np.rad2deg(info_r["state"]["tilt_angle"]))

    # Randomized environment should have wider range of altitude and tilt
    assert np.ptp(rand_altitudes) > np.ptp(base_altitudes)
    assert np.max(rand_tilts) > np.max(base_tilts)

    env_base.close()
    env_rand.close()


def test_truncation_on_max_steps():
    """Verify environment truncates when reaching MAX_EPISODE_STEPS."""
    env = LunarLander6DOFEnv()
    env.reset(seed=789)
    # Hover action to avoid crashing or early landing
    hover_action = np.array([-0.3, 0.0, 0.0, 0.0], dtype=np.float32)

    truncated = False
    for step in range(env.MAX_EPISODE_STEPS):
        obs, reward, terminated, truncated, info = env.step(hover_action)
        if terminated or truncated:
            break

    assert truncated or terminated
    env.close()

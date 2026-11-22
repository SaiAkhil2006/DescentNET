"""
Evaluation and Comparative Benchmarking script for LunarPilot.
Evaluates both the Baseline agent and the Randomized-trained agent on identical,
controlled test scenarios to scientifically measure robustness.
"""

import os
import argparse
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.lunar_env import LunarLander6DOFEnv
from environment.randomized_env import RandomizedLunarLander6DOFEnv
from simulation.physics import euler_to_quat, quat_to_euler, tilt_angle_from_upright
from agent.config import (
    BASELINE_DIR,
    RANDOMIZED_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
)


def generate_benchmark_test_scenarios(n_scenarios: int = 60, seed: int = 999) -> List[Dict[str, np.ndarray]]:
    """
    Generates a deterministic, standardized suite of test scenarios spanning 3 difficulty tiers:
    - Tier 1 (Nominal): Mild altitude (11m), small lateral offset (<1m), small tilt (<3 deg)
    - Tier 2 (Moderate): Altitude (12-14m), lateral offset (1-3m), tilt (5-12 deg), velocity (0.5-1.5 m/s)
    - Tier 3 (Severe / Disturbed): Altitude (10-15m), lateral offset (3-5m), tilt (15-22 deg), tumble (0.15-0.3 rad/s)
    """
    rng = np.random.default_rng(seed)
    scenarios = []

    n_per_tier = n_scenarios // 3
    # Tier 1: Nominal
    for i in range(n_per_tier):
        pos = np.array([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(11.0, 12.0)])
        roll = np.deg2rad(rng.uniform(-3.0, 3.0))
        pitch = np.deg2rad(rng.uniform(-3.0, 3.0))
        yaw = np.deg2rad(rng.uniform(-20.0, 20.0))
        quat = euler_to_quat(roll, pitch, yaw)
        lin_vel = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-1.0, -0.4)])
        ang_vel = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)])
        scenarios.append({
            "tier": "Nominal",
            "pos": pos,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
        })

    # Tier 2: Moderate
    for i in range(n_per_tier):
        pos = np.array([rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0), rng.uniform(12.0, 14.0)])
        roll = np.deg2rad(rng.uniform(-12.0, 12.0))
        pitch = np.deg2rad(rng.uniform(-12.0, 12.0))
        yaw = np.deg2rad(rng.uniform(-90.0, 90.0))
        quat = euler_to_quat(roll, pitch, yaw)
        lin_vel = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-1.2, 1.2), rng.uniform(-1.8, 0.0)])
        ang_vel = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15)])
        scenarios.append({
            "tier": "Moderate",
            "pos": pos,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
        })

    # Tier 3: Severe
    for i in range(n_scenarios - 2 * n_per_tier):
        pos = np.array([rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0), rng.uniform(10.0, 15.5)])
        roll = np.deg2rad(rng.uniform(-22.0, 22.0))
        pitch = np.deg2rad(rng.uniform(-22.0, 22.0))
        yaw = np.deg2rad(rng.uniform(-180.0, 180.0))
        quat = euler_to_quat(roll, pitch, yaw)
        lin_vel = np.array([rng.uniform(-2.2, 2.2), rng.uniform(-2.2, 2.2), rng.uniform(-2.5, 0.5)])
        ang_vel = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35)])
        scenarios.append({
            "tier": "Severe",
            "pos": pos,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
        })

    return scenarios


def evaluate_policy_on_scenarios(
    model: PPO,
    vec_norm: VecNormalize,
    scenarios: List[Dict[str, np.ndarray]],
    agent_name: str = "Agent",
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Evaluates a policy on a given list of scenarios, collecting trajectory metrics.
    """
    env = LunarLander6DOFEnv()
    records = []
    trajectories = []

    for idx, scen in enumerate(scenarios):
        # Reset environment with custom initial condition
        env.reset()
        state = env.sim.reset(
            pos=scen["pos"].copy(),
            quat=scen["quat"].copy(),
            lin_vel=scen["lin_vel"].copy(),
            ang_vel=scen["ang_vel"].copy(),
        )
        env._prev_dist = float(np.linalg.norm(state["pos"] - np.array([0.0, 0.0, env.Z_LANDING_PAD])))
        env._prev_tilt = state["tilt_angle"]
        obs = env._get_obs(state)

        episode_reward = 0.0
        step_count = 0
        done = False

        traj_pos = [state["pos"].copy()]
        traj_vel = [state["lin_vel"].copy()]
        traj_tilt = [np.rad2deg(state["tilt_angle"])]
        traj_thrust = []
        traj_time = [0.0]

        while not done:
            # Normalize observation using the training environment statistics
            norm_obs = vec_norm.normalize_obs(obs)
            action, _ = model.predict(norm_obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            curr_state = info["state"]
            traj_pos.append(curr_state["pos"].copy())
            traj_vel.append(curr_state["lin_vel"].copy())
            traj_tilt.append(np.rad2deg(curr_state["tilt_angle"]))
            traj_thrust.append(float(action[0] + 1.0) * 0.5 * env.sim.max_thrust)
            traj_time.append(curr_state["time"])

        final_state = info["state"]
        is_success = info.get("is_success", False)
        is_crashed = info.get("is_crashed", False)
        is_timeout = truncated and not (is_success or is_crashed)

        pos_xy_err = float(np.sqrt(final_state["pos"][0]**2 + final_state["pos"][1]**2))
        alt_err = float(abs(final_state["pos"][2] - env.Z_LANDING_PAD))
        final_vel_xy = float(np.sqrt(final_state["lin_vel"][0]**2 + final_state["lin_vel"][1]**2))
        final_vel_z = float(abs(final_state["lin_vel"][2]))
        final_tilt_deg = float(np.rad2deg(final_state["tilt_angle"]))
        final_ang_speed = float(np.linalg.norm(final_state["ang_vel"]))

        record = {
            "agent": agent_name,
            "scenario_id": idx,
            "tier": scen["tier"],
            "success": is_success,
            "crashed": is_crashed,
            "timeout": is_timeout,
            "reward": episode_reward,
            "steps": step_count,
            "flight_time_s": step_count * env.sim.dt_control,
            "pos_err_xy_m": pos_xy_err,
            "alt_err_m": alt_err,
            "vel_xy_ms": final_vel_xy,
            "vel_z_ms": final_vel_z,
            "tilt_deg": final_tilt_deg,
            "ang_speed_rads": final_ang_speed,
            "touchdown_feet": final_state["n_feet_in_contact"],
        }
        records.append(record)

        trajectories.append({
            "agent": agent_name,
            "scenario_id": idx,
            "tier": scen["tier"],
            "success": is_success,
            "crashed": is_crashed,
            "pos": np.array(traj_pos),
            "vel": np.array(traj_vel),
            "tilt": np.array(traj_tilt),
            "thrust": np.array(traj_thrust),
            "time": np.array(traj_time),
        })

    env.close()
    return pd.DataFrame(records), trajectories


def run_comparative_evaluation(
    n_scenarios: int = 60,
    seed: int = 999,
    baseline_model_path: str = None,
    baseline_vec_path: str = None,
    randomized_model_path: str = None,
    randomized_vec_path: str = None,
) -> pd.DataFrame:
    """
    Loads both agents, runs them against the benchmark test scenarios, and writes results/metrics.csv.
    """
    if baseline_model_path is None:
        baseline_model_path = os.path.join(BASELINE_DIR, "ppo_lunar_baseline_final.zip")
    if baseline_vec_path is None:
        baseline_vec_path = os.path.join(BASELINE_DIR, "vec_normalize_baseline.pkl")
    if randomized_model_path is None:
        randomized_model_path = os.path.join(RANDOMIZED_DIR, "ppo_lunar_randomized_final.zip")
    if randomized_vec_path is None:
        randomized_vec_path = os.path.join(RANDOMIZED_DIR, "vec_normalize_randomized.pkl")

    print("\n🔍 Loading Benchmark Test Scenarios...")
    scenarios = generate_benchmark_test_scenarios(n_scenarios=n_scenarios, seed=seed)
    print(f"   Created {len(scenarios)} scenarios ({len(scenarios)//3} per tier: Nominal, Moderate, Severe)")

    # 1. Evaluate Baseline Agent
    print(f"\n📦 Loading Baseline Model: {baseline_model_path}")
    dummy_env_b = DummyVecEnv([lambda: LunarLander6DOFEnv()])
    vec_norm_b = VecNormalize.load(baseline_vec_path, dummy_env_b)
    vec_norm_b.training = False
    model_b = PPO.load(baseline_model_path)

    print("🤖 Evaluating Baseline Agent across all benchmark scenarios...")
    df_baseline, trajs_b = evaluate_policy_on_scenarios(
        model_b, vec_norm_b, scenarios, agent_name="Baseline PPO"
    )

    # 2. Evaluate Randomized Agent
    print(f"\n📦 Loading Randomized Model: {randomized_model_path}")
    dummy_env_r = DummyVecEnv([lambda: RandomizedLunarLander6DOFEnv()])
    vec_norm_r = VecNormalize.load(randomized_vec_path, dummy_env_r)
    vec_norm_r.training = False
    model_r = PPO.load(randomized_model_path)

    print("🤖 Evaluating Randomized Agent across all benchmark scenarios...")
    df_randomized, trajs_r = evaluate_policy_on_scenarios(
        model_r, vec_norm_r, scenarios, agent_name="Randomized PPO"
    )

    # Combine results
    df_all = pd.concat([df_baseline, df_randomized], ignore_index=True)
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df_all.to_csv(metrics_csv_path, index=False)
    print(f"\n📊 Detailed evaluation results saved to: {metrics_csv_path}")

    # Compute Comparative Metrics Summary Table
    print("\n" + "=" * 90)
    print("🎯 LUNARPILOT 6-DOF COMPARATIVE PERFORMANCE SUMMARY")
    print("=" * 90)

    summary_rows = []
    for agent_name in ["Baseline PPO", "Randomized PPO"]:
        agent_df = df_all[df_all["agent"] == agent_name]
        
        # Overall
        succ_rate = agent_df["success"].mean() * 100.0
        crash_rate = agent_df["crashed"].mean() * 100.0
        mean_reward = agent_df["reward"].mean()
        mean_pos_err = agent_df["pos_err_xy_m"].mean()
        mean_vz = agent_df["vel_z_ms"].mean()
        mean_tilt = agent_df["tilt_deg"].mean()

        # Per tier success
        t1_succ = agent_df[agent_df["tier"] == "Nominal"]["success"].mean() * 100.0
        t2_succ = agent_df[agent_df["tier"] == "Moderate"]["success"].mean() * 100.0
        t3_succ = agent_df[agent_df["tier"] == "Severe"]["success"].mean() * 100.0

        summary_rows.append({
            "Agent": agent_name,
            "Success Rate (%)": f"{succ_rate:.1f}%",
            "Crash Rate (%)": f"{crash_rate:.1f}%",
            "Mean Reward": f"{mean_reward:.1f}",
            "Position Error (m)": f"{mean_pos_err:.2f}",
            "Touchdown Vz (m/s)": f"{mean_vz:.2f}",
            "Touchdown Tilt (deg)": f"{mean_tilt:.1f}°",
            "Nominal Tier Succ": f"{t1_succ:.1f}%",
            "Moderate Tier Succ": f"{t2_succ:.1f}%",
            "Severe Tier Succ": f"{t3_succ:.1f}%",
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("=" * 90 + "\n")

    return df_all, trajs_b, trajs_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs Randomized PPO on 6-DOF Lunar Lander")
    parser.add_argument("--n-scenarios", type=int, default=60, help="Number of benchmark evaluation scenarios (default: 60)")
    parser.add_argument("--seed", type=int, default=999, help="Random seed for test scenario generation (default: 999)")
    args = parser.parse_args()

    run_comparative_evaluation(n_scenarios=args.n_scenarios, seed=args.seed)

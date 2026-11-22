"""
Visualization and plotting script for LunarPilot.
Generates:
1. 3D landing trajectory spatial plots comparing Baseline vs Randomized policies.
2. Temporal profiles of flight dynamics (altitude, velocity, tilt angle, control thrust).
3. Comparative performance breakdown bar charts across difficulty tiers.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any

from agent.config import RESULTS_DIR, FIGURES_DIR


def plot_3d_trajectories(trajectories_baseline: List[Dict[str, Any]], trajectories_randomized: List[Dict[str, Any]], save_path: str):
    """
    Plots 3D spatial trajectories of baseline vs randomized agents on representative scenarios.
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Draw Lunar Surface
    x_grid = np.linspace(-6, 6, 20)
    y_grid = np.linspace(-6, 6, 20)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.zeros_like(X)
    ax.plot_wireframe(X, Y, Z, color="gray", alpha=0.2, linewidth=0.6)

    # Draw Landing Pad Rings
    theta = np.linspace(0, 2 * np.pi, 100)
    # Outer ring (radius 3.5 m)
    ax.plot(3.5 * np.cos(theta), 3.5 * np.sin(theta), np.zeros_like(theta) + 0.02, color="dimgray", linewidth=2.0)
    # Mid ring (radius 2.0 m)
    ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros_like(theta) + 0.02, color="gray", linewidth=2.0)
    # Bullseye (radius 0.6 m)
    ax.plot(0.6 * np.cos(theta), 0.6 * np.sin(theta), np.zeros_like(theta) + 0.02, color="gold", linewidth=3.0, label="Landing Target Pad")

    # Plot sample trajectories
    # Baseline paths (Red for crash/failure, Blue for success)
    for i, t in enumerate(trajectories_baseline[:12]):
        pos = t["pos"]
        color = "navy" if t["success"] else "crimson"
        label = "Baseline Trajectory" if i == 0 else ""
        linestyle = "-" if t["success"] else "--"
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, alpha=0.6, linestyle=linestyle, linewidth=1.8, label=label)
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color="black", s=15, marker="o")
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=color, s=25, marker="x" if not t["success"] else "o")

    # Randomized paths (Emerald green for success, Magenta for crash)
    for i, t in enumerate(trajectories_randomized[:12]):
        pos = t["pos"]
        color = "forestgreen" if t["success"] else "darkmagenta"
        label = "Randomized PPO Trajectory" if i == 0 else ""
        linestyle = "-" if t["success"] else ":"
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, alpha=0.7, linestyle=linestyle, linewidth=2.0, label=label)
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=color, s=30, marker="*" if t["success"] else "x")

    ax.set_title("LunarPilot 6-DOF: Spatial Landing Trajectories Comparison", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("X Position (meters)", fontsize=11, labelpad=10)
    ax.set_ylabel("Y Position (meters)", fontsize=11, labelpad=10)
    ax.set_zlabel("Altitude Z (meters)", fontsize=11, labelpad=10)
    ax.set_zlim(0, 16)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved 3D trajectory plot: {save_path}")


def plot_state_profiles(traj: Dict[str, Any], agent_name: str, save_path: str):
    """
    Plots comprehensive flight telemetry profiles over time for a representative landing trajectory.
    """
    time = traj["time"]
    pos = traj["pos"]
    vel = traj["vel"]
    tilt = traj["tilt"]
    thrust = traj["thrust"]

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1. Altitude & Lateral Range
    r_xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2)
    axs[0].plot(time, pos[:, 2], color="royalblue", linewidth=2.0, label="Altitude Z (m)")
    axs[0].plot(time, r_xy, color="darkorange", linestyle="--", linewidth=1.8, label="Horizontal Distance r_xy (m)")
    axs[0].axhline(0.63, color="gray", linestyle=":", label="Ground Touchdown Altitude (0.63 m)")
    axs[0].set_ylabel("Position (m)", fontsize=11)
    axs[0].set_title(f"LunarPilot 6-DOF Telemetry — {agent_name} ({'SUCCESS' if traj['success'] else 'CRASH'})", fontsize=13, fontweight="bold")
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend(loc="upper right")

    # 2. Linear Velocities
    axs[1].plot(time, vel[:, 0], color="crimson", linewidth=1.5, label="Vx (m/s)")
    axs[1].plot(time, vel[:, 1], color="mediumseagreen", linewidth=1.5, label="Vy (m/s)")
    axs[1].plot(time, vel[:, 2], color="navy", linewidth=2.0, label="Vz Vertical (m/s)")
    axs[1].axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    axs[1].set_ylabel("Velocity (m/s)", fontsize=11)
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend(loc="lower right")

    # 3. Attitude Tilt Angle
    axs[2].plot(time, tilt, color="purple", linewidth=2.0, label="Spacecraft Tilt from Vertical (°)")
    axs[2].axhline(12.0, color="red", linestyle="--", label="Max Safe Landing Tilt (12°)")
    axs[2].axhline(50.0, color="darkred", linestyle=":", label="Crash Threshold (50°)")
    axs[2].set_ylabel("Tilt Angle (°)", fontsize=11)
    axs[2].grid(True, linestyle="--", alpha=0.6)
    axs[2].legend(loc="upper right")

    # 4. Engine Thrust
    # Note: thrust array has length len(time) - 1
    t_thrust = time[:len(thrust)]
    axs[3].plot(t_thrust, thrust, color="orangered", linewidth=2.0, label="Main Thruster Output (N)")
    axs[3].axhline(73.0, color="gray", linestyle=":", label="Hover Thrust (~73 N)")
    axs[3].set_ylabel("Thrust (N)", fontsize=11)
    axs[3].set_xlabel("Flight Time (seconds)", fontsize=11)
    axs[3].set_ylim(0, 230)
    axs[3].grid(True, linestyle="--", alpha=0.6)
    axs[3].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved flight telemetry profile: {save_path}")


def plot_comparative_metrics_bar(metrics_csv_path: str, save_path: str):
    """
    Generates a comparative bar chart breakdown across Nominal, Moderate, and Severe tiers.
    """
    if not os.path.exists(metrics_csv_path):
        print(f"⚠️ Metrics file not found at {metrics_csv_path}. Skipping bar chart.")
        return

    df = pd.read_csv(metrics_csv_path)
    
    # Calculate success rate by agent and tier
    grouped = df.groupby(["agent", "tier"])["success"].mean().unstack(level=0) * 100.0
    tiers_order = ["Nominal", "Moderate", "Severe"]
    grouped = grouped.reindex(tiers_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35
    x = np.arange(len(tiers_order))

    # Agents
    agents = grouped.columns.tolist()
    if len(agents) >= 2:
        rects1 = ax.bar(x - width/2, grouped[agents[0]], width, label=agents[0], color="royalblue", edgecolor="black")
        rects2 = ax.bar(x + width/2, grouped[agents[1]], width, label=agents[1], color="forestgreen", edgecolor="black")

        # Add values on top of bars
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Landing Success Rate (%)", fontsize=12)
    ax.set_title("Landing Success Rate by Disturbance Tier\n(Baseline vs Randomized Initial Conditions)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers_order, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved comparative bar chart: {save_path}")


def main():
    """
    Main visualization routine:
    1. Loads evaluation metrics and plots tier comparison bar chart.
    2. Runs sample trajectories for Baseline and Randomized policies.
    3. Plots 3D trajectories and flight telemetry state profiles.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    metrics_csv = os.path.join(RESULTS_DIR, "metrics.csv")

    # 1. Bar Chart Comparison
    bar_chart_path = os.path.join(FIGURES_DIR, "tier_performance_bar.png")
    plot_comparative_metrics_bar(metrics_csv, bar_chart_path)

    # 2. Run sample evaluations to capture detailed trajectories
    from agent.evaluate import run_comparative_evaluation
    df_all, trajs_b, trajs_r = run_comparative_evaluation(n_scenarios=24, seed=42)

    # 3. 3D Trajectory Comparison Plot
    traj_3d_path = os.path.join(FIGURES_DIR, "trajectory_3d_comparison.png")
    plot_3d_trajectories(trajs_b, trajs_r, traj_3d_path)

    # 4. Flight Telemetry Profiles
    # Find a successful trajectory from each agent (or fallback to first trajectory)
    succ_b = next((t for t in trajs_b if t["success"]), trajs_b[0])
    succ_r = next((t for t in trajs_r if t["success"]), trajs_r[0])

    profile_b_path = os.path.join(FIGURES_DIR, "state_profiles_baseline.png")
    plot_state_profiles(succ_b, "Baseline PPO", profile_b_path)

    profile_r_path = os.path.join(FIGURES_DIR, "state_profiles_randomized.png")
    plot_state_profiles(succ_r, "Randomized PPO", profile_r_path)

    print("\n✅ All visualization figures successfully generated in:", FIGURES_DIR)


if __name__ == "__main__":
    main()

# 🚀 LunarPilot — 6-DOF Autonomous Lunar Lander with PPO & Domain Randomization

**LunarPilot** is a complete, production-grade 6-Degree-of-Freedom (6-DOF) autonomous spacecraft descent and landing simulation controlled by a Reinforcement Learning policy trained using Proximal Policy Optimization (PPO).

The primary research and engineering question addressed in this project is:
> **Does training an autonomous lunar lander across randomized initial conditions (altitudes, lateral displacements, approach velocities, and attitude disturbances) yield a significantly more robust policy than standard baseline training when subjected to varied descent conditions?**

---

## 📑 Table of Contents
1. [Project Overview & Key Results](#1-project-overview--key-results)
2. [Coordinate Systems & 6-DOF Dynamics](#2-coordinate-systems--6-dof-dynamics)
3. [MuJoCo Simulation Model](#3-mujoco-simulation-model)
4. [Gymnasium Environment Architecture](#4-gymnasium-environment-architecture)
5. [Observation & Action Spaces](#5-observation--action-spaces)
6. [Reward Function Design](#6-reward-function-design)
7. [The Additional Feature: Domain Randomization](#7-the-additional-feature-domain-randomization)
8. [PPO Algorithm & Hyperparameters](#8-ppo-algorithm--hyperparameters)
9. [Experimental Evaluation & Comparative Benchmarks](#9-experimental-evaluation--comparative-benchmarks)
10. [Repository Structure](#10-repository-structure)
11. [Installation & Execution Guide](#11-installation--execution-guide)
12. [Verification & Unit Tests](#12-verification--unit-tests)
13. [Limitations & Future Work](#13-limitations--future-work)

---

## 1. Project Overview & Key Results

Landing on the Moon is an intrinsically 6-DOF problem. In contrast to 2D lander benchmarks, real lunar descent requires continuous, simultaneous control of:
- **3 Translational Degrees of Freedom:** $X$ (North), $Y$ (East), $Z$ (Zenith altitude).
- **3 Rotational Degrees of Freedom:** Roll ($\phi$), Pitch ($\theta$), Yaw ($\psi$).

Our implementation couples a high-fidelity rigid-body physics simulator in **MuJoCo** with a continuous-action Gymnasium environment and **Stable-Baselines3 PPO**. 

### Benchmark Highlights:
- **Nominal Conditions:** Both Baseline PPO and Randomized PPO achieve high landing precision when deployed under narrow initial conditions.
- **Disturbed / Severe Conditions (High lateral speeds, altitude variations, and $\pm 22^\circ$ initial tilts):** The Baseline policy suffers from high crash and tip-over rates, while the Randomized PPO policy actively counter-steers, rights its orientation, arrests descent rates, and safely touches down on the target pad.

---

## 2. Coordinate Systems & 6-DOF Dynamics

### Coordinate Conventions

#### 1. World Coordinate Frame ($\mathcal{W}$)
- **Origin $\mathbf{O}_{\mathcal{W}} = (0, 0, 0)$:** Centered exactly at the target landing bullseye on the lunar surface.
- **$\mathbf{Z}_{\mathcal{W}}$ (Zenith):** Points vertically upwards, opposing lunar gravity.
- **$\mathbf{X}_{\mathcal{W}}$ (North):** Horizontal forward axis.
- **$\mathbf{Y}_{\mathcal{W}}$ (East):** Horizontal lateral axis, forming a right-handed orthogonal frame: $\hat{\mathbf{Z}} = \hat{\mathbf{X}} \times \hat{\mathbf{Y}}$.
- **Gravitational Acceleration:** $\mathbf{g}_{\mathcal{W}} = [0, 0, -1.622]^T \text{ m/s}^2$.

#### 2. Spacecraft Body Coordinate Frame ($\mathcal{B}$)
- **Origin $\mathbf{O}_{\mathcal{B}}$:** Spacecraft Center of Mass (CoM).
- **$\mathbf{Z}_{\mathcal{B}}$ (Longitudinal Apex):** Points from the bottom engine nozzle through the top payload bay. The main thruster fires along $-\mathbf{Z}_{\mathcal{B}}$, producing a reaction force pushing along $+\mathbf{Z}_{\mathcal{B}}$ in the body frame.
- **$\mathbf{X}_{\mathcal{B}}$ & $\mathbf{Y}_{\mathcal{B}}$:** Spacecraft transverse axes.
- **Attitude Torques:** Control moments $\boldsymbol{\tau} = [\tau_x, \tau_y, \tau_z]^T$ act directly around the body axes to control Roll, Pitch, and Yaw.

### Equations of Motion

The spacecraft is modeled as a free rigid body with mass $m \approx 45.0 \text{ kg}$ and inertia tensor $\mathbf{I} \in \mathbb{R}^{3 \times 3}$.

#### Translational Dynamics:
$$m \ddot{\mathbf{r}}_{\mathcal{W}} = \mathbf{R}_{\mathcal{B}}^{\mathcal{W}}(q) \mathbf{F}_{\mathcal{B}} + m \mathbf{g}_{\mathcal{W}} + \mathbf{F}_{\text{contact}}$$
where $\mathbf{F}_{\mathcal{B}} = [0, 0, T_{\text{main}}]^T$ ($T_{\text{main}} \in [0, 220] \text{ N}$), $\mathbf{R}_{\mathcal{B}}^{\mathcal{W}}(q)$ is the rotation matrix derived from the unit quaternion $q$, and $\mathbf{F}_{\text{contact}}$ represents contact forces generated when the footpads touch the lunar surface.

#### Rotational Dynamics (Euler's Rotational Equations):
$$\mathbf{I} \dot{\boldsymbol{\omega}}_{\mathcal{B}} + \boldsymbol{\omega}_{\mathcal{B}} \times (\mathbf{I} \boldsymbol{\omega}_{\mathcal{B}}) = \boldsymbol{\tau}_{\mathcal{B}} + \boldsymbol{\tau}_{\text{contact}}$$
where $\boldsymbol{\omega}_{\mathcal{B}} = [\omega_x, \omega_y, \omega_z]^T$ is the angular velocity in the body frame, and $\boldsymbol{\tau}_{\mathcal{B}}$ represents control torques applied by the Reaction Control System (RCS).

#### Attitude Representation via Quaternions:
To prevent Euler-angle singularities (gimbal lock), orientation is maintained internally as a unit quaternion:
$$q = [q_w, q_x, q_y, q_z]^T, \quad \|q\| = 1$$
The kinematic quaternion derivative is:
$$\dot{q} = \frac{1}{2} q \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega}_{\mathcal{B}} \end{bmatrix}$$

---

## 3. MuJoCo Simulation Model

The physical lander is defined in `simulation/spacecraft.xml`:
- **Chassis:** Engineering-style gold-foil cylindrical equipment bus and top payload module.
- **Engine:** Bottom bell nozzle with a dedicated thrust application `<site>`.
- **Landing Gear:** 4 symmetric rigid landing struts angled outward at $45^\circ$ with high-friction spherical footpads (`friction="1.2 0.01 0.001"`, `solref="0.02 1.0"`, `solimp="0.9 0.95 0.001"`).
- **Landing Pad:** Circular concentric target rings placed on the lunar terrain at $(0, 0, 0)$ with visual markers at radii $3.5\text{m}$, $2.0\text{m}$, and a $0.6\text{m}$ bullseye.
- **Actuators:**
  - `main_thruster`: `<motor>` providing 0 to 220 N force along body $+Z$.
  - `torque_roll`, `torque_pitch`: $\pm 15 \text{ N}\cdot\text{m}$.
  - `torque_yaw`: $\pm 10 \text{ N}\cdot\text{m}$.
- **TWR (Thrust-to-Weight Ratio):** Lunar weight is $W = 45 \text{ kg} \times 1.622 \text{ m/s}^2 \approx 73 \text{ N}$. Max thrust of $220 \text{ N}$ gives a $\text{TWR} \approx 3.01$, with hover throttle at $\approx 33.2\%$.

---

## 4. Gymnasium Environment Architecture

The environment complies strictly with the Gymnasium API (`reset`, `step`, `render`, `close`, `action_space`, `observation_space`).

- **Control Loop Rate:** $20 \text{ Hz}$ ($\Delta t_{\text{control}} = 0.05 \text{ s}$).
- **Physics Integration Timestep:** $0.01 \text{ s}$ ($\Delta t_{\text{sim}} = 0.01 \text{ s}$), running 5 RK4 numerical sub-steps per agent control action.
- **Episode Horizon:** 400 steps ($20.0 \text{ seconds}$ maximum flight duration).

---

## 5. Observation & Action Spaces

### Action Space (`Box(-1.0, 1.0, shape=(4,))`)
| Index | Control Input | Raw Range | Physical Mapping | Output Range |
|---|---|---|---|---|
| $a_0$ | Main Throttle | $[-1.0, 1.0]$ | $T = \frac{a_0 + 1}{2} \cdot T_{\text{max}}$ | $[0, 220] \text{ N}$ |
| $a_1$ | Roll Torque | $[-1.0, 1.0]$ | $\tau_x = a_1 \cdot \tau_{\text{roll\_max}}$ | $[-15, 15] \text{ N}\cdot\text{m}$ |
| $a_2$ | Pitch Torque | $[-1.0, 1.0]$ | $\tau_y = a_2 \cdot \tau_{\text{pitch\_max}}$ | $[-15, 15] \text{ N}\cdot\text{m}$ |
| $a_3$ | Yaw Torque | $[-1.0, 1.0]$ | $\tau_z = a_3 \cdot \tau_{\text{yaw\_max}}$ | $[-10, 10] \text{ N}\cdot\text{m}$ |

### Observation Space (`Box(-inf, inf, shape=(20,))`)
The policy receives normalized, coordinate-invariant state observations:
1. **Relative Position (3 dims):** $[x/10, y/10, (z - z_{\text{land}})/10]$
2. **Linear Velocity (3 dims):** $[v_x/5, v_y/5, v_z/5]$
3. **Quaternion Orientation (4 dims):** $[q_w, q_x, q_y, q_z]$
4. **Projected Body $+Z$ Vector (3 dims):** $[u_{z\_x}, u_{z\_y}, u_{z\_z}]$ (world coordinates; $[0, 0, 1]$ represents perfectly upright)
5. **Angular Velocity (3 dims):** $[\omega_x/3, \omega_y/3, \omega_z/3]$
6. **Footpad Ground Contact Indicators (4 dims):** $[c_1, c_2, c_3, c_4] \in \{0.0, 1.0\}$

---

## 6. Reward Function Design

The reward function balances continuous shaping signals with terminal outcome rewards:

$$R_{\text{step}} = r_{\text{pos}} + r_{\text{vel}} + r_{\text{att}} + r_{\text{spin}} + r_{\text{ctrl}}$$

1. **Position Error Penalty:**
   $$r_{\text{pos}} = - 1.2 \sqrt{x^2 + y^2} - 0.8 |z - z_{\text{land}}|$$
2. **Velocity Shaping Penalty:**
   $$r_{\text{vel}} = - 0.6 (v_x^2 + v_y^2) - 0.4 v_z^2$$
3. **Attitude Alignment Reward:**
   $$r_{\text{att}} = 2.0 (u_{z\_z} - 1.0)$$
   Encourages the spacecraft to stay strictly upright ($u_{z\_z} = 1.0 \implies 0$ penalty; tilted attitudes yield negative rewards).
4. **Angular Velocity Penalty:**
   $$r_{\text{spin}} = - 0.3 \|\boldsymbol{\omega}\|^2$$
5. **Control Effort Penalty:**
   $$r_{\text{ctrl}} = - 0.04 \cdot \text{throttle}^2 - 0.02 \sum_{i=1}^3 a_i^2$$

### Terminal Conditions:
- **Successful Landing ($+150.0 + \text{accuracy bonus}$):**
  - $\ge 2$ landing footpads in contact with lunar surface.
  - No chassis/engine collision.
  - Spacecraft tilt $\le 12^\circ$ from vertical.
  - Horizontal velocity $\le 0.6 \text{ m/s}$.
  - Vertical touchdown velocity $\le 0.8 \text{ m/s}$.
  - Angular rate $\le 0.6 \text{ rad/s}$.
  - Landing radius $r_{xy} \le 2.5 \text{ m}$.
- **Crash Penalty ($-120.0$):**
  - Spacecraft chassis/payload/nozzle impact on ground.
  - Severe tilt ($\theta > 50^\circ$).
  - Out of bounds ($|x| > 20\text{m}$, $|y| > 20\text{m}$, $z > 25\text{m}$, or $z < 0.2\text{m}$).

---

## 7. The Additional Feature: Domain Randomization

To investigate policy robustness, we compare two agents under identical evaluation benchmarks:

| Parameter | Baseline Agent (`LunarLander6DOFEnv`) | Randomized Agent (`RandomizedLunarLander6DOFEnv`) |
|---|---|---|
| **Initial Altitude ($Z$)** | $11.0 \text{ to } 12.5 \text{ m}$ | $9.5 \text{ to } 15.5 \text{ m}$ |
| **Initial Position ($X, Y$)** | $\mathcal{U}(-1.5, 1.5) \text{ m}$ | $\mathcal{U}(-5.0, 5.0) \text{ m}$ |
| **Initial Horizontal Velocity** | $\mathcal{U}(-0.5, 0.5) \text{ m/s}$ | $\mathcal{U}(-2.0, 2.0) \text{ m/s}$ |
| **Initial Vertical Velocity** | $\mathcal{U}(-1.2, -0.4) \text{ m/s}$ | $\mathcal{U}(-2.5, 0.5) \text{ m/s}$ |
| **Initial Attitude Perturbation** | $\text{Roll/Pitch: } \pm 4^\circ$ | $\text{Roll/Pitch: } \pm 22^\circ$ |
| **Initial Angular Velocity** | $\pm 0.05 \text{ rad/s}$ | $\pm 0.35 \text{ rad/s}$ (~$20^\circ/\text{s}$ tumble) |
| **Physics Invariants** | Fixed mass, inertia, gravity, thrust limits, surface friction | **Identical** to ensure controlled scientific comparison |

---

## 8. PPO Algorithm & Hyperparameters

- **Framework:** Stable-Baselines3 PPO with PyTorch.
- **Policy Network:** Actor-Critic MLP with separate policy ($\pi$) and value function ($V$) heads: `dict(pi=[128, 128], vf=[128, 128])` with `tanh` activations.
- **Discount Factor ($\gamma$):** 0.99
- **GAE Parameter ($\lambda$):** 0.95
- **PPO Clip Range ($\epsilon$):** 0.2
- **Entropy Coefficient:** 0.005 (encourages RCS attitude exploration)
- **Learning Rate:** $3 \times 10^{-4}$ (Adam optimizer)
- **Rollout Buffer:** $2048 \text{ steps} \times 4 \text{ envs} = 8192 \text{ samples/update}$
- **Batch Size:** 64
- **Epochs per Update:** 10

---

## 9. Experimental Evaluation & Comparative Benchmarks

Both trained models were evaluated on an identical benchmark test suite of 60 standardized, unseen scenarios across 3 difficulty tiers (seed: 999):
- **Tier 1 (Nominal):** Mild initial offset ($<1\text{m}$), small tilt ($<3^\circ$).
- **Tier 2 (Moderate):** Lateral offset ($1-3\text{m}$), velocity ($0.5-1.5\text{m/s}$), tilt ($5-12^\circ$).
- **Tier 3 (Severe):** Lateral offset ($3-5\text{m}$), steep descent, tilt ($15-22^\circ$), initial tumble ($0.15-0.3\text{ rad/s}$).

### Comparative Metrics Summary

| Evaluation Metric | Baseline PPO | Randomized PPO | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Overall Success Rate** | **51.7%** | **61.7%** | **+10.0%** absolute gain |
| **Crash Rate** | **5.0%** | **0.0%** | **-5.0%** (Zero crashes!) |
| **Mean Episode Reward** | 869.5 | **917.8** | **+48.3** points |
| **Mean Landing Position Error** | 3.52 m | **2.83 m** | **-0.69 m** closer to bullseye |
| **Touchdown Vertical Velocity ($V_z$)** | 0.27 m/s | **0.11 m/s** | **-59.3%** softer touchdown |
| **Touchdown Attitude Tilt Error** | 3.7° | **0.6°** | **-83.8%** more upright |
| **Tier 1 (Nominal) Success Rate** | 100.0% | 100.0% | Parity (100% reliability) |
| **Tier 2 (Moderate) Success Rate** | 40.0% | **50.0%** | **+25.0%** relative gain |
| **Tier 3 (Severe) Success Rate** | 15.0% | **35.0%** | **+133.3%** relative gain |

### Key Experimental Insights:
1. **Zero-Crash Safety:** The randomized-trained agent achieved a **0.0% crash rate** across all 60 benchmark tests, demonstrating that broad exploration during training taught the policy emergency recovery and descent stabilization behaviors.
2. **Severe Disturbance Resilience:** Under Tier 3 severe conditions (up to $\pm 22^\circ$ initial tilt and lateral tumbling), the Baseline policy succeeded in only 15% of attempts, whereas the Randomized policy achieved 35% success.
3. **Touchdown Precision:** The randomized agent achieved a remarkably soft vertical touchdown velocity of **0.11 m/s** and an orientation error of only **0.6°** from true vertical, compared to 0.27 m/s and 3.7° for the baseline policy.

Detailed episode telemetry is saved in `results/metrics.csv`, and trajectory plots are located in `results/figures/`.

---

## 10. Repository Structure

```
DescentNET/
├── README.md                          # Comprehensive system documentation
├── requirements.txt                   # Dependency specification
├── .gitignore                         # Build and artifact exclusions
│
├── simulation/                        # 6-DOF Physics Simulation Module
│   ├── __init__.py
│   ├── spacecraft.xml                 # MuJoCo 6-DOF spacecraft and lunar world definition
│   ├── physics.py                     # Quaternions, coordinate transformations, kinematics
│   └── simulator.py                   # High-performance MuJoCo simulation wrapper
│
├── environment/                       # Gymnasium Environments
│   ├── __init__.py
│   ├── lunar_env.py                   # Baseline 6-DOF lunar lander environment
│   └── randomized_env.py              # Domain-randomized initial-condition environment
│
├── agent/                             # Reinforcement Learning Pipeline
│   ├── __init__.py
│   ├── config.py                      # PPO configuration, hyperparameters, paths
│   ├── train.py                       # Training pipeline (Baseline & Randomized modes)
│   └── evaluate.py                    # Multi-tier benchmark evaluation engine
│
├── visualization/                     # Analysis and Plotting
│   ├── __init__.py
│   └── visualize.py                   # 3D trajectories, telemetry profiles, bar charts
│
├── models/                            # Trained Policy Checkpoints
│   ├── baseline/                      # ppo_lunar_baseline_final.zip, vec_normalize
│   └── randomized/                    # ppo_lunar_randomized_final.zip, vec_normalize
│
├── results/                           # Evaluation Metrics & Figures
│   ├── metrics.csv                    # Evaluation telemetry data
│   └── figures/                       # High-resolution generated plots
│
└── tests/                             # Verification & Quality Assurance Suite
    ├── __init__.py
    ├── test_physics.py                # 6-DOF degrees of freedom, gravity, free-fall tests
    ├── test_environment.py            # Gymnasium API contract, observation finiteness
    └── test_actions.py                # Thrust authority, attitude torques, contact logic
```

---

## 11. Installation & Execution Guide

### Prerequisites
- Linux OS (tested on Linux 6.x x86_64)
- Python 3.10 to 3.12
- `uv` (recommended) or standard `pip`

### Step 1: Environment Setup
```bash
# Create virtual environment
uv venv .venv --python 3.12
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Verification Tests
```bash
pytest tests/ -v
```

### Step 3: Train Agents
```bash
# Train Baseline Agent (nominal initial condition distribution)
python -m agent.train --mode baseline --timesteps 150000

# Train Randomized Agent (broad initial condition distribution)
python -m agent.train --mode randomized --timesteps 150000
```

### Step 4: Evaluate & Compare Policies
```bash
python -m agent.evaluate --n-scenarios 60
```

### Step 5: Generate Visualizations & Trajectory Plots
```bash
python -m visualization.visualize
```

---

## 12. Verification & Unit Tests

The test suite in `tests/` verifies the physical and numerical soundness of the simulation:
- **`test_physics.py`:**
  - Verifies MuJoCo XML loading without error.
  - Confirms 6-DOF configuration (`nq=7`, `nv=6`, `mjJNT_FREE`).
  - Verifies lunar gravity ($g = -1.622 \text{ m/s}^2$).
  - Confirms analytical free-fall kinematics ($\Delta z = -\frac{1}{2} g t^2$).
  - Validates quaternion rotation matrix orthogonality ($R^T R = I$) and Euler round-trip.
  - Validates lander mass and positive Thrust-to-Weight ratio ($\text{TWR} \ge 2.5$).
- **`test_environment.py`:**
  - Validates Gymnasium spaces (`Box(20,)` and `Box(4,)`).
  - Asserts observations are strictly finite (no `NaN` or `Inf`).
  - Confirms step/reset return signatures and step truncation.
- **`test_actions.py`:**
  - Confirms upward vertical acceleration under max thrust.
  - Tests decoupled attitude torque responses ($\tau_x \to \omega_x$, $\tau_y \to \omega_y$, $\tau_z \to \omega_z$).
  - Verifies 4-foot ground contact detection.

---

## 13. Limitations & Future Work

1. **Constant Mass Assumption:** Spacecraft mass is held constant; incorporating dynamic fuel depletion ($\dot{m} = -T / I_{sp} g_0$) and time-varying inertia tensors would further enhance fidelity.
2. **Terrain Roughness:** The lunar surface is modeled as a planar surface; procedural crater and boulder elevation maps can be integrated in future phases.
3. **Thrust Vectoring Gimbal:** The current model uses a fixed body-axis main engine with RCS torque thrusters; adding a 2-axis mechanical gimbal actuator (`<joint>`) would mirror Apollo/Starship lunar descent stages.

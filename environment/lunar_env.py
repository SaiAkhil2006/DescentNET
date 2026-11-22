"""
6-DOF Lunar Lander Gymnasium Environment (Baseline).
Complies with Gymnasium API (reset, step, render, close, observation_space, action_space).
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.simulator import LanderSimulator
from simulation.physics import euler_to_quat, tilt_angle_from_upright


class LunarLander6DOFEnv(gym.Env):
    """
    Gymnasium environment for a 6-DOF autonomous lunar lander.

    Coordinate Frame:
      World frame origin (0, 0, 0) is at the center of the target landing pad on the surface.
      +Z is zenith (upwards).
      At touchdown, the lander footpads rest on Z = 0, so the center of mass sits at Z_LANDING_PAD (~0.63 m).
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    # Center of mass height above lunar ground when resting flat on all 4 footpads
    Z_LANDING_PAD = 0.63
    MAX_EPISODE_STEPS = 400  # 20 seconds at 20 Hz

    def __init__(
        self,
        xml_path: Optional[str] = None,
        control_freq: float = 20.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.control_freq = control_freq

        # Initialize underlying MuJoCo 6-DOF simulator
        self.sim = LanderSimulator(xml_path=xml_path, control_freq=control_freq)

        # Action Space: 4 Continuous bounded inputs in [-1.0, 1.0]
        # a[0]: Main Thruster throttle [-1, 1] -> [0, max_thrust (220 N)]
        # a[1]: Roll torque command [-1, 1] -> [-15, 15] N*m
        # a[2]: Pitch torque command [-1, 1] -> [-15, 15] N*m
        # a[3]: Yaw torque command [-1, 1] -> [-10, 10] N*m
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation Space: 20 continuous dimensions
        # [0:3]   Relative position [x/10, y/10, (z - z_land)/10]
        # [3:6]   Linear velocity [vx/5, vy/5, vz/5]
        # [6:10]  Quaternion orientation [qw, qx, qy, qz]
        # [10:13] Projected body +Z axis in world [uz_x, uz_y, uz_z]
        # [13:16] Angular velocity [wx/3, wy/3, wz/3]
        # [16:20] 4 Footpad contact indicators [c1, c2, c3, c4] (0.0 or 1.0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        self._step_count = 0
        self._prev_dist = 0.0
        self._prev_tilt = 0.0

    def _get_obs(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Construct the normalized 20-dimensional observation vector.
        """
        pos = state["pos"]
        rel_pos = np.array([
            pos[0] / 10.0,
            pos[1] / 10.0,
            (pos[2] - self.Z_LANDING_PAD) / 10.0,
        ], dtype=np.float32)

        lin_vel = np.array(state["lin_vel"] / 5.0, dtype=np.float32)
        quat = np.array(state["quat"], dtype=np.float32)
        body_z = np.array(state["body_z_world"], dtype=np.float32)
        ang_vel = np.array(state["ang_vel"] / 3.0, dtype=np.float32)
        foot_contacts = np.array(state["foot_contacts"], dtype=np.float32)

        obs = np.concatenate([rel_pos, lin_vel, quat, body_z, ang_vel, foot_contacts])
        return obs.astype(np.float32)

    def _sample_initial_conditions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Baseline initial condition distribution:
        - Drops from ~11-12 m altitude with small horizontal offset and near-upright attitude.
        """
        # Position: z ~ 11.5 +/- 0.5m, x,y ~ U(-1.5, 1.5)
        pos = np.array([
            self.np_random.uniform(-1.5, 1.5),
            self.np_random.uniform(-1.5, 1.5),
            self.np_random.uniform(11.0, 12.5),
        ], dtype=np.float64)

        # Small initial Euler perturbations: roll/pitch +/- 4 deg, yaw +/- 15 deg
        roll = np.deg2rad(self.np_random.uniform(-4.0, 4.0))
        pitch = np.deg2rad(self.np_random.uniform(-4.0, 4.0))
        yaw = np.deg2rad(self.np_random.uniform(-15.0, 15.0))
        quat = euler_to_quat(roll, pitch, yaw)

        # Velocities: moderate downward descent, minor lateral drift
        lin_vel = np.array([
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(-1.2, -0.4),
        ], dtype=np.float64)

        # Angular velocity: small initial rate
        ang_vel = np.array([
            self.np_random.uniform(-0.05, 0.05),
            self.np_random.uniform(-0.05, 0.05),
            self.np_random.uniform(-0.05, 0.05),
        ], dtype=np.float64)

        return pos, quat, lin_vel, ang_vel

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        pos, quat, lin_vel, ang_vel = self._sample_initial_conditions()
        state = self.sim.reset(pos=pos, quat=quat, lin_vel=lin_vel, ang_vel=ang_vel)

        self._prev_dist = float(np.linalg.norm(state["pos"] - np.array([0.0, 0.0, self.Z_LANDING_PAD])))
        self._prev_tilt = state["tilt_angle"]

        obs = self._get_obs(state)
        info = {
            "is_success": False,
            "is_crashed": False,
            "state": state,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Action mapping:
        # a[0] in [-1, 1] -> thrust in [0, max_thrust]
        throttle = float(np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0))
        thrust = throttle * self.sim.max_thrust

        # Attitude torques
        roll_torque = float(action[1]) * self.sim.max_torque_roll
        pitch_torque = float(action[2]) * self.sim.max_torque_pitch
        yaw_torque = float(action[3]) * self.sim.max_torque_yaw
        torques = np.array([roll_torque, pitch_torque, yaw_torque], dtype=np.float64)

        # Step physics simulator
        state = self.sim.step(thrust=thrust, torques=torques)

        # Evaluate rewards and termination conditions
        reward, terminated, truncated, info = self._compute_reward_and_termination(state, action, throttle)

        obs = self._get_obs(state)
        info["state"] = state
        info["step"] = self._step_count

        return obs, reward, terminated, truncated, info

    def _compute_reward_and_termination(
        self, state: Dict[str, Any], action: np.ndarray, throttle: float
    ) -> Tuple[float, bool, bool, Dict[str, Any]]:
        pos = state["pos"]
        lin_vel = state["lin_vel"]
        ang_vel = state["ang_vel"]
        body_z_world = state["body_z_world"]
        tilt = state["tilt_angle"]
        foot_contacts = state["foot_contacts"]
        n_feet = state["n_feet_in_contact"]
        chassis_contact = state["chassis_contact"]

        # Geometric distances
        r_xy = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))
        h_z = float(max(0.0, pos[2] - self.Z_LANDING_PAD))
        dist_3d = float(np.sqrt(r_xy ** 2 + h_z ** 2))

        speed_xy = float(np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2))
        speed_z = float(abs(lin_vel[2]))
        speed_3d = float(np.linalg.norm(lin_vel))
        ang_speed = float(np.linalg.norm(ang_vel))

        # -------------------------------------------------------------
        # Continuous Shaping Rewards (Potential-Based & Stabilization)
        # -------------------------------------------------------------
        # 1. Potential-based progress towards landing target (preserves policy invariance)
        r_progress = 15.0 * (self._prev_dist - dist_3d)
        self._prev_dist = dist_3d

        # 2. Upright attitude reward: positive reward for maintaining vertical orientation (+1.0 when upright)
        # body_z_world[2] = cos(tilt)
        r_attitude = 1.0 * float(body_z_world[2])

        # 3. Rotational stability: penalize excessive angular tumbling
        r_ang_vel = - 0.1 * float(ang_speed ** 2)

        # 4. Velocity shaping: guide vertical descent towards -0.8 m/s (or -0.3 m/s near ground) and suppress drift
        target_vz = -0.8 if h_z > 1.5 else -0.3
        r_vel_z = - 0.25 * float((lin_vel[2] - target_vz) ** 2)
        r_vel_xy = - 0.25 * float(speed_xy ** 2)

        # 5. Footpad ground contact guidance (reward placing landing gear on surface)
        r_contact = 1.0 * float(n_feet)

        # 6. Control effort penalty: promote fuel conservation and smooth RCS commands
        r_effort = - 0.02 * (throttle ** 2) - 0.01 * float(np.sum(action[1:] ** 2))

        step_reward = r_progress + r_attitude + r_ang_vel + r_vel_z + r_vel_xy + r_contact + r_effort

        # -------------------------------------------------------------
        # Termination & Touchdown Checks
        # -------------------------------------------------------------
        terminated = False
        truncated = False
        is_success = False
        is_crashed = False

        # Criteria for Crash / Failure:
        # A) Chassis/body/nozzle impacts the surface
        # B) Extreme tilt (angle > 65 degrees)
        # C) Drifts outside the operational arena boundaries
        out_of_bounds = (
            abs(pos[0]) > 20.0
            or abs(pos[1]) > 20.0
            or pos[2] > 25.0
            or pos[2] < 0.2
        )
        excessive_tilt = tilt > np.deg2rad(65.0)

        if chassis_contact or excessive_tilt or out_of_bounds:
            terminated = True
            is_crashed = True
            step_reward -= 250.0  # Decisive penalty to eliminate suicide exploitation
            info = {
                "is_success": False,
                "is_crashed": True,
                "crash_reason": (
                    "chassis_contact" if chassis_contact else
                    "excessive_tilt" if excessive_tilt else "out_of_bounds"
                ),
            }
            return step_reward, terminated, truncated, info

        # Criteria for Successful Landing:
        # 1) At least 2 legs firmly in contact with ground
        # 2) Upright attitude: tilt <= 15 degrees
        # 3) Low horizontal speed: speed_xy <= 0.8 m/s
        # 4) Low vertical touchdown speed: speed_z <= 1.0 m/s
        # 5) Low angular velocity: ang_speed <= 0.8 rad/s
        # 6) Close to landing bullseye: r_xy <= 3.0 m
        stable_orientation = tilt <= np.deg2rad(15.0)
        gentle_touchdown = speed_xy <= 0.8 and speed_z <= 1.0 and ang_speed <= 0.8
        on_target = r_xy <= 3.0

        if n_feet >= 2 and stable_orientation and gentle_touchdown and on_target:
            terminated = True
            is_success = True
            # Scaled success bonus depending on landing accuracy
            accuracy_bonus = max(0.0, 1.0 - (r_xy / 3.0)) * 100.0
            step_reward += 300.0 + accuracy_bonus
            info = {
                "is_success": True,
                "is_crashed": False,
                "accuracy_bonus": accuracy_bonus,
            }
            return step_reward, terminated, truncated, info

        # Check episode step limit truncation
        if self._step_count >= self.MAX_EPISODE_STEPS:
            truncated = True

        info = {
            "is_success": is_success,
            "is_crashed": is_crashed,
        }
        return step_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.sim.render(mode="rgb_array")
        return None

    def close(self):
        self.sim.close()

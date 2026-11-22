"""
Randomized Initial-Conditions 6-DOF Lunar Lander Gymnasium Environment.
Implements our key project improvement: Training with broad initial state distributions
to foster robust landing policies across varied altitudes, lateral offsets, and initial tilts.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np

from environment.lunar_env import LunarLander6DOFEnv
from simulation.physics import euler_to_quat


class RandomizedLunarLander6DOFEnv(LunarLander6DOFEnv):
    """
    Subclass of LunarLander6DOFEnv with broadened initial conditions.
    
    Physics parameters (mass, inertia, gravity, thrust limits, surface) remain identical
    to maintain a scientifically valid controlled experimental comparison.
    Only the initial starting distribution is widened.
    """

    def _sample_initial_conditions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Broadened initial-condition distribution:
        - Wider altitude range: z0 in [9.5, 15.5] m
        - Wider horizontal footprint: x0, y0 in [-5.0, 5.0] m
        - Significant initial horizontal velocities: vx, vy in [-2.0, 2.0] m/s
        - Significant vertical velocities: vz in [-2.5, 0.5] m/s (can start descending quickly)
        - Substantial initial attitude perturbance: roll, pitch in [-22 deg, +22 deg]
        - Initial spin/tumble: angular velocities in [-0.35, 0.35] rad/s (~20 deg/s)
        """
        # Position
        pos = np.array([
            self.np_random.uniform(-5.0, 5.0),
            self.np_random.uniform(-5.0, 5.0),
            self.np_random.uniform(9.5, 15.5),
        ], dtype=np.float64)

        # Orientation: significant roll, pitch, and arbitrary yaw
        roll = np.deg2rad(self.np_random.uniform(-22.0, 22.0))
        pitch = np.deg2rad(self.np_random.uniform(-22.0, 22.0))
        yaw = np.deg2rad(self.np_random.uniform(-180.0, 180.0))
        quat = euler_to_quat(roll, pitch, yaw)

        # Linear velocities: horizontal drift and steep descent
        lin_vel = np.array([
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-2.0, 2.0),
            self.np_random.uniform(-2.5, 0.5),
        ], dtype=np.float64)

        # Angular velocities: non-trivial initial tumble rates
        ang_vel = np.array([
            self.np_random.uniform(-0.35, 0.35),
            self.np_random.uniform(-0.35, 0.35),
            self.np_random.uniform(-0.35, 0.35),
        ], dtype=np.float64)

        return pos, quat, lin_vel, ang_vel

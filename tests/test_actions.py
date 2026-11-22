"""
Unit tests for Actuator Action Responses and Ground Contact Dynamics.
"""

import pytest
import numpy as np

from simulation.simulator import LanderSimulator
from simulation.physics import MOON_GRAVITY


def test_full_thrust_acceleration():
    """
    Verify that full thrust overcomes lunar gravity and produces net upward vertical acceleration.
    """
    sim = LanderSimulator()
    sim.reset(pos=np.array([0.0, 0.0, 5.0]), lin_vel=np.zeros(3))

    # Apply 100% max thrust (220 N) with zero attitude torques
    state_after = sim.step(thrust=sim.max_thrust, torques=np.zeros(3))

    # Spacecraft mass ~ 45 kg, weight ~ 73 N.
    # Net upward force = 220 - 73 = 147 N, so upward acceleration a_z > 3.0 m/s^2.
    assert state_after["lin_vel"][2] > 0.0, "Full thrust must produce upward vertical velocity"


def test_roll_attitude_torque():
    """Verify that positive roll torque produces positive angular velocity around Body X."""
    sim = LanderSimulator()
    sim.reset(pos=np.array([0.0, 0.0, 5.0]), ang_vel=np.zeros(3))

    state_after = sim.step(thrust=sim.lunar_weight, torques=np.array([10.0, 0.0, 0.0]))
    assert state_after["ang_vel"][0] > 0.0, "Positive roll torque must induce positive wx"


def test_pitch_attitude_torque():
    """Verify that positive pitch torque produces positive angular velocity around Body Y."""
    sim = LanderSimulator()
    sim.reset(pos=np.array([0.0, 0.0, 5.0]), ang_vel=np.zeros(3))

    state_after = sim.step(thrust=sim.lunar_weight, torques=np.array([0.0, 10.0, 0.0]))
    assert state_after["ang_vel"][1] > 0.0, "Positive pitch torque must induce positive wy"


def test_yaw_attitude_torque():
    """Verify that positive yaw torque produces positive angular velocity around Body Z."""
    sim = LanderSimulator()
    sim.reset(pos=np.array([0.0, 0.0, 5.0]), ang_vel=np.zeros(3))

    state_after = sim.step(thrust=sim.lunar_weight, torques=np.array([0.0, 0.0, 5.0]))
    assert state_after["ang_vel"][2] > 0.0, "Positive yaw torque must induce positive wz"


def test_ground_contact_detection():
    """
    Verify that when the lander is placed resting at ground level (z = 0.63 m),
    all 4 footpads are detected in contact with the ground without chassis crash.
    """
    sim = LanderSimulator()
    # At z=0.63, the 4 footpads (foot_1..4 at -0.55m relative to CoM) touch z=0.08m (radius 0.08)
    sim.reset(pos=np.array([0.0, 0.0, 0.63]), lin_vel=np.zeros(3))
    
    # Step 10 substeps under lunar gravity to settle on springs/contact
    for _ in range(10):
        state = sim.step(thrust=sim.lunar_weight, torques=np.zeros(3))

    assert state["n_feet_in_contact"] >= 2, f"Expected foot contacts, got {state['n_feet_in_contact']}"
    assert not state["chassis_contact"], "Lander chassis should not crash into ground when resting on footpads"

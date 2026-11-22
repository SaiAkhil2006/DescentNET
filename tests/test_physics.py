"""
Unit tests for 6-DOF Rigid-Body Dynamics, Coordinate Frames, and MuJoCo Model.
"""

import os
import pytest
import numpy as np
import mujoco

from simulation.simulator import LanderSimulator
from simulation.physics import (
    MOON_GRAVITY,
    quat_to_rot_mat,
    quat_to_euler,
    euler_to_quat,
    rotate_vector_by_quat,
    get_body_z_in_world,
    tilt_angle_from_upright,
)


def test_mujoco_model_loads_successfully():
    """Verify that spacecraft.xml loads without errors or warnings."""
    sim = LanderSimulator()
    assert sim.model is not None
    assert sim.data is not None
    assert sim.lander_body_id >= 0
    assert sim.lander_joint_id >= 0


def test_full_6_dof_configuration():
    """
    Verify that the spacecraft has 6 degrees of freedom:
    Freejoint must have 7 position coordinates (3 translation + 4 quaternion)
    and 6 velocity coordinates (3 linear + 3 angular).
    """
    sim = LanderSimulator()
    # Check joint type is free (mjJNT_FREE = 0)
    jnt_type = sim.model.jnt_type[sim.lander_joint_id]
    assert jnt_type == mujoco.mjtJoint.mjJNT_FREE

    # 7 generalized positions (qpos) and 6 generalized velocities (qvel)
    assert sim.model.nq == 7
    assert sim.model.nv == 6


def test_lunar_gravity():
    """Verify that lunar gravity is configured to -1.622 m/s^2 along world -Z."""
    sim = LanderSimulator()
    gravity = sim.model.opt.gravity
    assert gravity[0] == 0.0
    assert gravity[1] == 0.0
    assert np.isclose(gravity[2], -MOON_GRAVITY, atol=1e-3)


def test_freefall_kinematics():
    """
    Verify unforced free-fall under lunar gravity.
    Spacecraft starts from rest at z=10.0m with 0 thrust.
    After time t, delta_z should equal -0.5 * g * t^2.
    """
    sim = LanderSimulator(control_freq=100.0)
    initial_z = 10.0
    state = sim.reset(pos=np.array([0.0, 0.0, initial_z]), lin_vel=np.zeros(3))
    
    t_flight = 0.5  # seconds (50 steps at 100 Hz)
    n_steps = int(t_flight * 100)
    for _ in range(n_steps):
        state = sim.step(thrust=0.0, torques=np.zeros(3))

    expected_z = initial_z - 0.5 * MOON_GRAVITY * (t_flight ** 2)
    actual_z = state["pos"][2]
    assert np.isclose(actual_z, expected_z, atol=0.01)

    expected_vz = - MOON_GRAVITY * t_flight
    actual_vz = state["lin_vel"][2]
    assert np.isclose(actual_vz, expected_vz, atol=0.01)


def test_quaternion_math():
    """Verify quaternion conversion, rotation matrix orthogonality, and Euler round-trip."""
    # Test identity quaternion
    q_id = np.array([1.0, 0.0, 0.0, 0.0])
    R_id = quat_to_rot_mat(q_id)
    assert np.allclose(R_id, np.eye(3))

    # Test arbitrary rotation
    roll_in, pitch_in, yaw_in = np.deg2rad(15.0), np.deg2rad(-20.0), np.deg2rad(45.0)
    q = euler_to_quat(roll_in, pitch_in, yaw_in)
    
    # Check unit norm
    assert np.isclose(np.linalg.norm(q), 1.0)

    # Check rotation matrix properties: R^T R = I and det(R) = 1
    R = quat_to_rot_mat(q)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    # Check roundtrip to Euler
    euler_out = quat_to_euler(q)
    assert np.allclose(euler_out, [roll_in, pitch_in, yaw_in], atol=1e-5)


def test_spacecraft_mass_and_thrust_capability():
    """
    Verify mass, lunar weight, and thrust authority.
    Maximum thrust must be sufficient to decelerate and overcome lunar weight.
    """
    sim = LanderSimulator()
    assert 20.0 <= sim.mass <= 60.0
    assert sim.lunar_weight > 0.0

    # Max thrust should exceed lunar weight with a safe TWR (TWR >= 2.0)
    twr = sim.max_thrust / sim.lunar_weight
    assert twr >= 2.0, f"Thrust to weight ratio too low: {twr:.2f}"

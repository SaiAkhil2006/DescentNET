"""
MuJoCo-based 6-DOF Spacecraft Simulator.
Wraps MuJoCo model, physics state integration, and low-level actuators.
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
import mujoco

from simulation.physics import (
    quat_to_rot_mat,
    quat_to_euler,
    get_body_z_in_world,
    tilt_angle_from_upright,
)


class LanderSimulator:
    """
    Simulates a 6-DOF Lunar Lander spacecraft in MuJoCo.

    Degrees of freedom:
      - Translation: x, y, z (meters)
      - Rotation: roll, pitch, yaw (represented via unit quaternion q = [qw, qx, qy, qz])
      - Generalized velocities: 3 linear velocities (vx, vy, vz), 3 angular velocities (wx, wy, wz)
    """

    def __init__(self, xml_path: Optional[str] = None, control_freq: float = 20.0):
        """
        Initialize the simulation.
        :param xml_path: Absolute or relative path to spacecraft.xml.
        :param control_freq: Frequency of the RL agent control step in Hz (default 20 Hz, dt_control = 0.05s).
        """
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), "spacecraft.xml")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")

        self.xml_path = os.path.abspath(xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.physics_dt = self.model.opt.timestep  # e.g., 0.01 s
        self.control_freq = control_freq
        self.dt_control = 1.0 / control_freq
        self.n_substeps = max(1, int(round(self.dt_control / self.physics_dt)))

        # Cache body, geom, and site IDs for high performance
        self.lander_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "lander")
        self.lander_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lander_freejoint")
        self.ground_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "lunar_ground")

        # Actuator IDs: 0: main_thruster, 1: torque_roll, 2: torque_pitch, 3: torque_yaw
        self.act_main = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "main_thruster")
        self.act_roll = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque_roll")
        self.act_pitch = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque_pitch")
        self.act_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "torque_yaw")

        # Foot geom IDs for landing contact detection
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{i}")
            for i in range(1, 5)
        ]
        
        # Body geoms that constitute a crash if they contact the ground
        self.crash_geom_names = ["main_bus", "payload_top", "engine_bell"]
        self.crash_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.crash_geom_names
        ]

        # Spacecraft mass and weight on Moon
        self.mass = float(self.model.body_subtreemass[self.lander_body_id])
        self.lunar_weight = self.mass * abs(float(self.model.opt.gravity[2]))

        # Max actuator limits
        self.max_thrust = float(self.model.actuator_ctrlrange[self.act_main, 1])
        self.max_torque_roll = float(self.model.actuator_ctrlrange[self.act_roll, 1])
        self.max_torque_pitch = float(self.model.actuator_ctrlrange[self.act_pitch, 1])
        self.max_torque_yaw = float(self.model.actuator_ctrlrange[self.act_yaw, 1])

        # Renderer placeholder (initialized lazily for visualization)
        self._renderer = None

    def reset(
        self,
        pos: np.ndarray = np.array([0.0, 0.0, 10.0]),
        quat: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0]),
        lin_vel: np.ndarray = np.array([0.0, 0.0, 0.0]),
        ang_vel: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> Dict[str, Any]:
        """
        Reset spacecraft to specified 6-DOF position, orientation, and velocities.
        """
        mujoco.mj_resetData(self.model, self.data)

        # qpos for freejoint has 7 entries: [x, y, z, qw, qx, qy, qz]
        quat_norm = quat / np.linalg.norm(quat)
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = quat_norm

        # qvel for freejoint has 6 entries: [vx, vy, vz, wx, wy, wz]
        self.data.qvel[0:3] = lin_vel
        self.data.qvel[3:6] = ang_vel

        # Clear control commands
        self.data.ctrl[:] = 0.0

        # Step forward 1 zero-step to update kinematics and contact dynamics
        mujoco.mj_forward(self.model, self.data)

        return self.get_state()

    def step(self, thrust: float, torques: np.ndarray) -> Dict[str, Any]:
        """
        Advance the simulation by one control step (n_substeps physics steps).
        :param thrust: Main engine thrust command (Newtons, in [0, max_thrust]).
        :param torques: 3D attitude torque vector [roll, pitch, yaw] in N*m.
        """
        # Clamp commands to physical actuator limits
        clamped_thrust = np.clip(thrust, 0.0, self.max_thrust)
        clamped_roll = np.clip(torques[0], -self.max_torque_roll, self.max_torque_roll)
        clamped_pitch = np.clip(torques[1], -self.max_torque_pitch, self.max_torque_pitch)
        clamped_yaw = np.clip(torques[2], -self.max_torque_yaw, self.max_torque_yaw)

        # Apply controls to MuJoCo actuator buffers
        self.data.ctrl[self.act_main] = clamped_thrust
        self.data.ctrl[self.act_roll] = clamped_roll
        self.data.ctrl[self.act_pitch] = clamped_pitch
        self.data.ctrl[self.act_yaw] = clamped_yaw

        # Integrate physics over n_substeps
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        """
        Extract complete 6-DOF state and contact information.
        """
        pos = self.data.qpos[0:3].copy()
        quat = self.data.qpos[3:7].copy()
        # Ensure quaternion has positive scalar part for continuity
        if quat[0] < 0:
            quat = -quat
        quat = quat / np.linalg.norm(quat)

        lin_vel = self.data.qvel[0:3].copy()
        ang_vel = self.data.qvel[3:6].copy()

        # Orientation metrics
        rot_mat = quat_to_rot_mat(quat)
        euler = quat_to_euler(quat)
        body_z_world = get_body_z_in_world(quat)
        tilt = tilt_angle_from_upright(quat)

        # Contact detection via MuJoCo contact pairs
        foot_contacts = [False, False, False, False]
        chassis_contact = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            # Check if one geom is ground
            if geom1 == self.ground_geom_id or geom2 == self.ground_geom_id:
                other_geom = geom2 if geom1 == self.ground_geom_id else geom1
                
                # Check footpads
                for foot_idx, f_id in enumerate(self.foot_geom_ids):
                    if other_geom == f_id:
                        foot_contacts[foot_idx] = True

                # Check fuselage / engine / chassis
                if other_geom in self.crash_geom_ids:
                    chassis_contact = True

        return {
            "time": float(self.data.time),
            "pos": pos,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
            "rot_mat": rot_mat,
            "euler": euler,
            "body_z_world": body_z_world,
            "tilt_angle": tilt,
            "foot_contacts": foot_contacts,
            "n_feet_in_contact": int(sum(foot_contacts)),
            "chassis_contact": chassis_contact,
        }

    def render(self, mode: str = "rgb_array", width: int = 640, height: int = 480) -> Optional[np.ndarray]:
        """
        Render the simulation view into an RGB numpy array.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

import pybullet as pb

import numpy as np
from math import pi
from pathlib import Path

from .arm import Arm
from .body import Body
from .chain import Chain
from .camera import Camera
from .arm_control import ArmPositionController
from .universal_robot import UR5Kinematics
from .gripper import RG6Gripper
from .gripper_control import RG6GripperController
from .robotiq_gripper import *


class PRLUR5Robot:
    def __init__(self, client_id, with_gripper=True, fixed=True):
        flags = (
            pb.URDF_USE_INERTIA_FROM_FILE
            | pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
            # | pb.URDF_USE_SELF_COLLISION
            # | pb.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )

        # Robot
        self._body = Body.load(
            "prl_ur5/robot.urdf",
            flags=flags,
            useFixedBase=fixed,
            client_id=client_id,
        )

        self.moving_joints_map = {}

        for i in range(pb.getNumJoints(self._body.body_id)):
            info = pb.getJointInfo(self._body.body_id, i)
            joint_name = info[1].decode("utf-8", "strict")
            joint_type = info[2]
            link_name = info[12].decode("utf-8", "strict")

            if joint_type != pb.JOINT_FIXED:
                self.moving_joints_map[joint_name] = i

        left_arm = Arm(self._body, tip_link_name="left_gripper_grasp_frame")
        left_arm.controller = ArmPositionController(left_arm, gains=0.1)
        left_arm._kinematics = UR5Kinematics(
            left_arm._chain,
            prefix="left_",
        )

        # We want the tip to be the camera frame
        right_arm = Arm(self._body, tip_link_name="right_camera_color_optical_frame")
        right_arm.controller = ArmPositionController(right_arm, gains=0.1)
        right_arm._kinematics = UR5Kinematics(
            right_arm._chain,
            prefix="right_",
        )

        gripper = None
        if with_gripper:
            gripper = RG6Gripper(self._body, prefix="left_")
            gripper.controller = RG6GripperController(gripper)

        self._arm = left_arm
        self._right_arm = right_arm
        self._gripper = gripper
        self.client_id = client_id
        self._wrist_cameras = []

    def enable_wrist_camera(self, prefix="right_", width=1280, height=720):
        link = self._body.link(f"{prefix}camera_color_optical_frame")
        cam = Camera(width, height, self._body.client_id)
        cam.attach(link=link, orn=(0, 0, np.pi))
        self._wrist_cameras.append(cam)

    def attach_wrist_camera(
        self, prefix="right_", pos=(0, 0, 0), orn=(0, 0, np.pi), width=1280, height=720
    ):
        link = self._body.link(f"{prefix}camera_color_optical_frame")
        cam = Camera(width, height, self._body.client_id)
        cam.attach(link=link, pos=pos, orn=orn)
        return cam

    @property
    def arm(self):
        return self._arm

    @property
    def right_arm(self):
        return self._right_arm

    @property
    def gripper(self):
        return self._gripper

    @property
    def wrist_cameras(self):
        return self._wrist_cameras

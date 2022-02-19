import numpy as np
import pybullet as pb  # only used for euler2quat

from math import pi
from enum import Enum
from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from mime.config import assets_path
from .utils import load_textures

OBJ_TEXTURES_PATH = assets_path() / "textures" / "objects" / "simple"


class Target(Enum):
    CUBE = 1
    CYLINDER = 2


class PushScene(TableScene):
    def __init__(self, **kwargs):
        super(PushScene, self).__init__(**kwargs)
        self._target = None
        self._marker = None

        # linear velocity x2 for the real setup
        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

        self._target_type = Target.CYLINDER

        if self._target_type == Target.CYLINDER:
            radius_range = {"low": 0.03, "high": 0.07}
            height_range = {"low": 0.05, "high": 0.09}
            self._cylinder_size_range = {
                "low": [radius_range["low"], height_range["low"]],
                "high": [radius_range["high"], height_range["high"]],
            }
        elif self._target_type == Target.CUBE:
            self._cube_size_range = {"low": 0.03, "high": 0.07}
        else:
            raise ValueError(f"Target Type {self._target_type} is not valid.")

        self._marker_size_range = {"low": 0.075, "high": 0.085}
        self._success_distance = 0.02

    def load(self, np_random):
        super(PushScene, self).load(np_random)

    def load_textures(self, np_random):
        super().load_textures(np_random)
        self._modder._textures["objects"] = load_textures(OBJ_TEXTURES_PATH, np_random)

    def reset(
        self,
        np_random,
        **kwargs,
    ):
        """
        Reset the target and arm position.
        """

        super(PushScene, self).reset(np_random)

        target_size = kwargs.get("target_size", None)
        target_type = kwargs.get("target_type", self._target_type)
        if type(target_type) is int:
            target_type = Target(target_type)

        target_position = kwargs.get("target_position", None)
        target_color = kwargs.get("target_color", None)
        marker_size = kwargs.get("marker_size", None)
        marker_position = kwargs.get("marker_position", None)
        marker_color = kwargs.get("marker_color", None)
        gripper_position = kwargs.get("gripper_position", None)

        modder = self._modder

        # load and randomize cage
        modder.load_cage(np_random)
        if self._domain_rand:
            modder.randomize_cage_visual(np_random)

        self._safe_height = [0.12, 0.15]
        self._object_workspace = [[-0.62, -0.15, 0.0], [-0.42, 0.22, 0.2]]
        self._marker_workspace = [[-0.35, -0.10, 0.0], [-0.22, 0.10, 0.2]]

        # define workspace, tool position and cylinder position
        low_object, high_object = self._object_workspace
        low_object, high_object = np.array(low_object.copy()), np.array(
            high_object.copy()
        )
        low_marker, high_marker = self._marker_workspace
        low_marker, high_marker = np.array(low_marker.copy()), np.array(
            high_marker.copy()
        )

        if self._target is not None:
            self._target.remove()

        if self._marker is not None:
            self._marker.remove()

        # load cube, set to random size and random position
        if target_type == Target.CUBE:
            if not target_size:
                target, target_size = modder.load_mesh(
                    "cube", self._cube_size_range, np_random
                )
            else:
                target, target_size = modder.load_mesh("cube", target_size, np_random)
            target_height = target_size
            target_width = target_size * 2
        elif target_type == Target.CYLINDER:
            if not target_size:
                target, target_size = modder.load_mesh(
                    "cylinder", self._cylinder_size_range, np_random
                )
            else:
                target, target_size = modder.load_mesh(
                    "cylinder", target_size, np_random
                )
            target_height = target_size[1]
            target_width = target_size[0] * 2
        else:
            raise ValueError(f"Target Type {self._target_type} is not valid.")

        self._target = target
        self._target_height = target_height
        self._target_width = target_width

        rand_color_target = False
        if target_color is None:
            target_color = np.array([27, 79, 119, 255], dtype=float) / 255
            rand_color_target = True

        if rand_color_target:
            modder.randomize_object_color(np_random, target, target_color)

        target.color = target_color
        low_object[:2] += self._target_width / 2
        high_object[:2] -= self._target_width / 2

        # Create Marker
        marker, marker_size = modder.add_marker(
            "square", self._marker_size_range, np_random
        )

        valid_conf = False
        while not valid_conf:
            if gripper_position is None:
                (
                    sampled_gripper_position,
                    gripper_orn,
                ) = self.random_gripper_pose(np_random)
            else:
                sampled_gripper_position = gripper_position
                gripper_orn = [pi, 0, pi / 2]

            if marker_position is None:
                sampled_marker_position = np_random.uniform(
                    low=low_marker, high=high_marker
                )
            else:
                sampled_marker_position = marker_position

            if target_position is None:
                sampled_target_position = np_random.uniform(
                    low=low_object, high=high_object
                )
            else:
                sampled_target_position = target_position

            if (
                np.linalg.norm(
                    sampled_marker_position[:2] - sampled_target_position[:2]
                )
                >= self._target_width * 2
            ) and np.linalg.norm(
                sampled_gripper_position[:2] - sampled_target_position[:2]
            ) >= self._target_width * 1.5:
                valid_conf = True
                if marker_position is None:
                    marker_position = sampled_marker_position
                if target_position is None:
                    target_position = sampled_target_position
                if gripper_position is None:
                    gripper_position = sampled_gripper_position

        q0 = self.robot.arm.controller.joints_target
        q = self.robot.arm.kinematics.inverse(gripper_position, gripper_orn, q0)
        self.robot.arm.reset(q)

        self._target_position = [
            target_position[0],
            target_position[1],
            self._target_height / 2,
        ]
        self._target.position = self._target_position

        marker_position[2] = 0.0001

        rand_color_marker = False
        if marker_color is None:
            marker_color = np.array([153, 51, 26, 255], dtype=float) / 255
            rand_color_marker = True
        self._marker = marker
        self._marker.position = marker_position
        self._marker_position = marker_position
        self._marker.color = marker_color
        # if self._domain_rand:
        if rand_color_marker:
            modder.randomize_object_color(np_random, marker, marker_color)

    def script(self):
        """
        Script to generate expert demonstrations.
        """
        arm = self.robot.arm
        grip = self.robot.gripper
        target_pos = np.array(self.target_position)
        marker_pos = np.array(self._marker_position)

        def_gripper_orn = [pi, 0, pi / 2]

        push_v = marker_pos[:2] - target_pos[:2]
        push_distance = np.linalg.norm(push_v)
        push_norm_v = push_v / push_distance
        ref_v = np.array([1, 0]) if push_norm_v[0] > 0 else np.array([-1, 0])
        push_angle = np.arccos(
            np.dot(ref_v, push_norm_v)
            / (np.linalg.norm(ref_v) * np.linalg.norm(push_norm_v))
        )

        push_angle = push_angle if push_norm_v[0] * push_norm_v[1] > 0 else -push_angle

        init_push_xy = target_pos[:2] - push_norm_v * self._target_width
        init_push_pos = np.array(
            [init_push_xy[0], init_push_xy[1], self._target_height]
        )

        init_push_orn = np.array(
            [
                def_gripper_orn[0],
                def_gripper_orn[1],
                def_gripper_orn[2] + push_angle,
            ]
        )

        sc = Script(self)
        end_push_xy = marker_pos[:2] - push_norm_v * (self._target_width / 2)
        end_push_pos = np.array([end_push_xy[0], end_push_xy[1], self._target_height])

        return [
            sc.grip_close(grip),
            sc.tool_move(arm, init_push_pos + [0, 0, 0.08], orn=init_push_orn),
            sc.tool_move(arm, init_push_pos),
            sc.tool_move(arm, end_push_pos),
        ]

    @property
    def target_position(self):
        pos_target, _ = self._target.position
        return np.array(pos_target)

    @property
    def marker_position(self):
        pos_marker, _ = self._marker.position
        return np.array(pos_marker)

    @property
    def target_orientation(self):
        _, orn_cube = self._target.position
        orn_cube_euler = pb.getEulerFromQuaternion(orn_cube)
        return np.array([orn_cube_euler[-1] / np.pi * 180])

    @property
    def distance_to_target(self):
        return np.subtract(self.marker_position, self.target_position)

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        return np.linalg.norm(self.distance_to_target[:2]) < self._success_distance


def test_scene():
    from time import sleep

    scene = PushScene(robot_type="PRL_UR5")
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        obs = scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()

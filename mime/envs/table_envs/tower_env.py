from .tower_scene import TowerScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv

import numpy as np


class TowerEnv(TableEnv):
    """Tower environment, trajectory observation, linear tool control"""

    def __init__(self, **kwargs):
        scene = TowerScene(**kwargs)
        super(TowerEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            "distance_to_target",
            "target_position",
            "linear_velocity",
            "grip_forces",
            "grip_width",
        )
        self.action_space = self._make_dict_space(
            "linear_velocity",
            # 'joint_velocity',
            "grip_velocity",
        )

    def _get_observation(self, scene):
        return dict(
            tool_position=scene.robot.arm.tool.state.position[0],
            cubes_position=scene.cubes_position,
            distance_to_cubes=scene.distance_to_cubes,
            linear_velocity=scene.robot.arm.tool.state.velocity[0],
            grip_state=scene.robot.gripper.controller.state,
        )


class TowerCamEnv(TableCamEnv):
    """Tower environment, camera observation, linear tool control"""

    def __init__(
        self,
        view_rand,
        gui_resolution,
        cam_resolution,
        crop_size,
        num_cameras,
        **kwargs,
    ):
        scene = TowerScene(**kwargs)
        super(TowerCamEnv, self).__init__(
            scene=scene,
            view_rand=view_rand,
            gui_resolution=gui_resolution,
            cam_resolution=cam_resolution,
            crop_size=crop_size,
            num_cameras=num_cameras,
        )

        self.action_space = self._make_dict_space(
            "linear_velocity",
            # 'joint_velocity',
            "grip_velocity",
        )

    def _get_observation(self, scene):
        obs = super()._get_observation(scene)
        for cam_name, cameras_list in self.cameras.items():
            for i, camera_info in enumerate(cameras_list):
                mask = obs[f"mask_{cam_name}{i}"]
                mask[mask >= self.scene.OBJECT_LABEL] = self.scene.OBJECT_LABEL

        return obs

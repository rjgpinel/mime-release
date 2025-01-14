from .pick_scene import PickScene
from .table_env import TableEnv
from .table_cam_env import TableCamEnv


class PickEnv(TableEnv):
    """Pick environment, trajectory observation, linear tool control"""

    def __init__(self, **kwargs):
        scene = PickScene(**kwargs)
        super(PickEnv, self).__init__(scene)

        self.observation_space = self._make_dict_space(
            "distance_to_target",
            "target_position",
            "linear_velocity",
            "grip_forces",
            "grip_width",
            "gripper_pose",
        )
        self.action_space = self._make_dict_space(
            "linear_velocity",
            # 'joint_velocity',
            "grip_velocity",
        )

    def _get_observation(self, scene):
        obs_dic = super(PickEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
                target_orientation=scene.target_orientation,
                gripper_pose=scene.gripper_pose,
            )
        )

        return obs_dic


class PickCamEnv(TableCamEnv):
    """Pick environment, camera observation, linear tool control"""

    def __init__(
        self,
        view_rand,
        gui_resolution,
        cam_resolution,
        crop_size,
        num_cameras,
        **kwargs
    ):
        scene = PickScene(**kwargs)
        super(PickCamEnv, self).__init__(
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
        obs_dic = super(PickCamEnv, self)._get_observation(scene)
        obs_dic.update(
            dict(
                distance_to_goal=scene.distance_to_target,
                target_position=scene.target_position,
                target_orientation=scene.target_orientation,
            )
        )

        return obs_dic

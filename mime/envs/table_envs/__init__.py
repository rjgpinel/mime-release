import itertools
from gym.envs.registration import register

from mime.envs.table_envs.utils import *
from mime.envs.table_envs.table_modder import *
from mime.envs.table_envs.pick_env import *
from mime.envs.table_envs.push_env import *
from mime.envs.table_envs.tower_env import *

# from mime.envs.table_envs.pour_env import *
# from mime.envs.table_envs.bowl_env import *
# from mime.envs.table_envs.rope_env import *


environments = {
    "Pick": dict(max_episode_steps=150),
    "Rope": dict(max_episode_steps=400),
    "Push": dict(max_episode_steps=1000),
    "Tower": dict(max_episode_steps=300),
    "Pour": dict(max_episode_steps=400),
    "Bowl": dict(max_episode_steps=600),
}

extra_env_args = {}

env_files = {}

robots = ["UR5", "PRL_UR5"]
horizons = [None, 1]
num_cams = [0, 1, 2, 5, 10, 15, 20]
rand_cam = ["Hard", "Easy", "FOV", ""]
rand_obj = ["VarSize", ""]
domain_rand = [False, True]
load_egl = [False, True]
env_combinations = list(
    itertools.product(
        robots, horizons, num_cams, rand_cam, domain_rand, rand_obj, load_egl
    )
)

cam_resolution = (640, 360)
gui_resolution = (1280, 720)
crop_size = 224

for env, kwargs in environments.items():
    orig_env_horizon = kwargs["max_episode_steps"]
    for (
        robot,
        horizon,
        num_cam,
        rand_cam,
        domain_rand,
        rand_obj,
        load_egl,
    ) in env_combinations:
        horizon_str, camera_str, dr_str, obj_str, egl_str = "", "", "", "", ""
        if horizon is not None:
            kwargs = dict(max_episode_steps=horizon)
            horizon_str = "Short"
        else:
            kwargs = dict(max_episode_steps=orig_env_horizon)
        if load_egl:
            egl_str = "EGL"
        if num_cam > 0:
            if num_cam == 1:
                camera_str = "{}RandCam".format(rand_cam) if rand_cam else "Cam"
            else:
                if not rand_cam:
                    # we don't want envs with multiple non-random cameras
                    continue
                camera_str = "{}{}RandCam".format(num_cam, rand_cam)
        if domain_rand:
            dr_str = "DR-"

        if rand_obj:
            obj_str = f"-{rand_obj}"

        # example of full name DR-UR5-ShortEGL-Pick5RandCam
        if horizon_str + egl_str != "":
            env_name = "{}{}-{}{}-{}{}{}".format(
                dr_str, robot, horizon_str, egl_str, env, camera_str, obj_str
            )
        else:
            env_name = "{}{}-{}{}{}".format(dr_str, robot, env, camera_str, obj_str)
        env_file = env_files[env] if env in env_files else env

        if num_cam == 0:
            if rand_cam or load_egl:
                # for no camera environments there is no difference
                # we don't want to reregister the environments
                continue
            env_args = dict(
                robot_type=robot,
                domain_rand=domain_rand,
                rand_obj=rand_obj,
            )
            if env in extra_env_args:
                env_args.update(extra_env_args[env])
            register(
                id="{}Env-v0".format(env_name),
                entry_point="mime.envs.table_envs:{}Env".format(env_file),
                kwargs=env_args,
                reward_threshold=2.0,
                **kwargs,
            )
        else:
            env_args = dict(
                robot_type=robot,
                view_rand=rand_cam,
                gui_resolution=gui_resolution,
                cam_resolution=cam_resolution,
                crop_size=crop_size,
                num_cameras=num_cam,
                rand_obj=rand_obj,
                domain_rand=domain_rand,
                load_egl=load_egl,
            )
            if env in extra_env_args:
                env_args.update(extra_env_args[env])
            register(
                id="{}Env-v0".format(env_name),
                entry_point="mime.envs.table_envs:{}CamEnv".format(env_file),
                kwargs=env_args,
                reward_threshold=2.0,
                **kwargs,
            )

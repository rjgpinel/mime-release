import torch
import click
import mime
import gym
import re

import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from einops import rearrange
from torchvision.transforms import InterpolationMode

from robos2r.core.tf import (
    pos_mat_from_vec,
    translation_from_matrix,
    quaternion_from_matrix,
)
from robos2r.data.lmdb import list_to_lmdb
from robos2r.config import data_path


def blend(im_a, im_b, alpha):
    im_a = Image.fromarray(im_a)
    im_b = Image.fromarray(im_b)
    im_blend = Image.blend(im_a, im_b, alpha).convert("RGB")
    im_blend = np.asanyarray(im_blend).copy()
    return im_blend


crop_size = 224
resize_crop_im = T.Compose(
    [
        T.Resize(crop_size, antialias=True),
        T.CenterCrop(crop_size),
    ]
)
resize_crop_seg = T.Compose(
    [
        T.Resize(crop_size, interpolation=InterpolationMode.NEAREST),
        T.CenterCrop(crop_size),
    ]
)


def process_obs(obs, cam_list):
    new_obs = []
    for cam_name in cam_list:
        new_obs.append(torch.tensor(obs[f"rgb_{cam_name}"]))
    new_obs = torch.stack(new_obs).float()
    new_obs = rearrange(new_obs, "b h w c -> b c h w")
    new_obs = resize_crop_im(new_obs)
    new_obs = rearrange(new_obs, "b c h w -> b h w c")
    new_obs = np.asarray(new_obs, dtype=np.uint8)
    new_obs_dic = {}
    for i, cam_name in enumerate(cam_list):
        new_obs_dic[f"rgb_{cam_name}"] = new_obs[i]
    return new_obs_dic


@click.command()
@click.option(
    "-p",
    "--path",
    default="sim2real/new_setup/",
    type=str,
)
@click.option("-o", "--output_path", default="sim2real/new_setup/", type=str)
@click.option("-db", "--debug/--no-debug", default=False, is_flag=True)
@click.option("-v", "--viz/--no-viz", default=False, is_flag=True)
def main(path, output_path, debug, viz):

    directory = Path(data_path()) / path
    output_directory = Path(data_path()) / output_path
    output_directory.mkdir(parents=True, exist_ok=True)

    data_files = []
    data_files.extend(directory.glob("*.pkl"))

    # Define cleaning parameters
    cube_label = 3
    min_cube_area = 150
    gripper_label = 0
    min_gripper_area = 200

    dataset = []
    for data_file in tqdm(data_files):
        with open(str(data_file), "rb") as f:
            data = pkl.load(f)
            cam_list = data["cam_list"]
            cams_info = data["cam_info"]
            try:
                dataset += data["dataset"]
            except:
                pass

    cams_info = {
        cam["camera_name"]: (cam["translation"], cam["rotation"]) for cam in cams_info
    }

    print(f"Processing dataset of size {len(dataset)}")

    env = gym.make("PRL_UR5-PickCamEnv-v0")
    env = env.unwrapped
    if debug:
        scene = env.unwrapped.scene
        scene.renders(True)

    clean_data_idx = []

    for scene_idx in tqdm(range(len(dataset))):
        scene = dataset[scene_idx]
        new_ims = process_obs(scene, cam_list)
        scene.update(new_ims)
        gripper_pose = scene["gripper_pose"]
        cube_pose = scene["target_position"], scene["target_orientation"]

        sim_obs = env.reset(
            cube_pose=cube_pose,
            gripper_pose=gripper_pose,
        )

        valid = True

        for cam_name in cam_list:
            shot = sim_obs

            im = sim_obs[f"rgb_{cam_name}0"]
            seg = sim_obs[f"mask_{cam_name}0"]
            im_real = scene[f"rgb_{cam_name}"]

            dataset[scene_idx][f"rgb_{cam_name}"] = im_real
            dataset[scene_idx][f"depth_{cam_name}"] = np.zeros((224, 224))
            dataset[scene_idx][f"camera_optical_frame_tf_{cam_name}"] = cams_info

            if viz:
                im_blend = blend(im, im_real, 0.5)

                plt.subplot(121)
                plt.imshow(im_real)
                plt.subplot(122)
                plt.imshow(im)
                plt.show()
                plt.imshow(im_blend)
                plt.show()

            area_cube = np.sum(seg == cube_label)
            area_gripper = np.sum(seg == gripper_label)
            if area_cube < min_cube_area or area_gripper < min_gripper_area:
                valid = False
                break

        if valid:
            clean_data_idx.append(scene_idx)

    print(f"Clean dataset with {len(clean_data_idx)} samples")
    train_idx = np.random.choice(
        clean_data_idx, int(0.8 * len(clean_data_idx)), replace=False
    )
    val_idx = [i for i in clean_data_idx if i not in train_idx]

    print(f"Saving train dataset with {len(train_idx)} samples.")
    train_data = []
    for scene_idx in train_idx:
        train_data.append(dataset[scene_idx])
    train_path = str(output_directory / "train.lmdb")
    list_to_lmdb(train_data, train_path)

    print(f"Saving val dataset with {len(val_idx)} samples.")
    val_data = []
    for scene_idx in val_idx:
        val_data.append(dataset[scene_idx])

    val_path = str(output_directory / "val.lmdb")
    list_to_lmdb(val_data, val_path)


if __name__ == "__main__":
    main()

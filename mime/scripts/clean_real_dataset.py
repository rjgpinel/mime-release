import click
import mime
import gym

import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import InterpolationMode

from robos2r.data.lmdb import list_to_lmdb


@click.command()
@click.option("-db", "--debug/--no-debug", default=False, is_flag=True)
def main(debug):

    directory = Path("/home/rgarciap/Datasets/sim2real/real/multicam/")
    data_files = []
    data_files.extend(directory.glob("*.pkl"))
    cube_label = 2

    dataset = []
    for data_file in data_files:
        with open(str(data_file), "rb") as f:
            try:
                dataset += pkl.load(f)
            except:
                pass

    print(f"Processing dataset of size {len(dataset)}")

    # TODO: How to pass it to camera env
    crop_size = 224
    resize_size = 224

    resize_im = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(resize_size),
        ]
    )
    resize_seg = T.Compose(
        [
            T.ToPILImage(),
            T.Resize(resize_size, interpolation=InterpolationMode.NEAREST),
        ]
    )

    crop_transform = T.CenterCrop(crop_size)

    env = gym.make("PRL_UR5-PickCamEnv-v0")
    if debug:
        scene = env.unwrapped.scene
        scene.renders(True)

    clean_data_idx = []

    for scene_idx in tqdm(range(len(dataset))):
        scene = dataset[scene_idx]
        scene_data = scene["obs"]
        cube_pose = scene["cube_pose"]

        for instance_idx in range(len(scene_data)):
            instance = scene_data[instance_idx]

            joint_state = instance["joint_state"]

            joints_qp = dict(
                zip(joint_state["joint_names"], joint_state["joint_position"])
            )
            right_arm_joints_qp = [
                joints_qp["right_shoulder_pan_joint"],
                joints_qp["right_shoulder_lift_joint"],
                joints_qp["right_elbow_joint"],
                joints_qp["right_wrist_1_joint"],
                joints_qp["right_wrist_2_joint"],
                joints_qp["right_wrist_3_joint"],
            ]

            gripper_pose = instance["tool_tf"]

            sim_obs = env.reset(
                cube_pose=cube_pose,
                gripper_pose=gripper_pose,
            )

            env.scene.robot.right_arm.reset(right_arm_joints_qp)
            camera = env.scene.robot._wrist_cameras[0]
            camera.shot()
            im = np.array(crop_transform(resize_im(camera.rgb)))
            seg = np.array(crop_transform(resize_seg(camera.mask)))
            im_real = np.array(crop_transform(resize_im(instance["rgb"][:, :, :3])))

            # plt.subplot(121)
            # plt.imshow(im_real)
            # plt.subplot(122)
            # plt.imshow(im)
            # plt.show()
            plt.imshow(seg)
            plt.show()
            area = np.sum(seg == cube_label)
            if area > 150:
                clean_data_idx.append((scene_idx, instance_idx))

    print(f"Clean dataset with {len(clean_data_idx)} samples")
    all_idx = list(range(len(clean_data_idx)))
    train_idx = np.random.choice(all_idx, int(0.8 * len(clean_data_idx)), replace=False)
    val_idx = [i for i in all_idx if i not in train_idx]

    print(f"Saving train dataset with {len(train_idx)} samples.")
    train_data = []
    for i in train_idx:
        scene_idx, instance_idx = clean_data_idx[i]
        data = process_instance_data(scene_idx, instance_idx, dataset)
        train_data.append(data)
    train_path = "/home/rgarciap/Datasets/sim2real/real/multicam/train.lmdb"
    list_to_lmdb(train_data, train_path)

    print(f"Saving val dataset with {len(val_idx)} samples.")
    val_data = []
    for i in val_idx:
        scene_idx, instance_idx = clean_data_idx[i]
        data = process_instance_data(scene_idx, instance_idx, dataset)
        val_data.append(data)
    val_path = "/home/rgarciap/Datasets/sim2real/real/multicam/val.lmdb"
    list_to_lmdb(val_data, val_path)


def process_instance_data(scene_idx, instance_idx, dataset):
    output = {}
    scene = dataset[scene_idx]
    instance_data = scene["obs"][instance_idx]
    cube_pose = scene["cube_pose"]
    output["cube_pose"] = cube_pose
    output.update(instance_data)
    return output


if __name__ == "__main__":
    main()

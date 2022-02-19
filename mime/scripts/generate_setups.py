import click
import mime
import gym

import yaml

import numpy as np
import pickle as pkl

from pathlib import Path
from mime.config import assets_path


@click.command()
@click.option(
    "-e",
    "--episodes",
    default=25,
    type=int,
)
@click.option(
    "-i",
    "--initial-episode",
    default=5000,
    type=int,
)
@click.option("-n", "--num-cubes", default=2, type=int)
@click.option(
    "-o",
    "--output-path",
    default="/home/rgarciap/episodes_info_tower/",
    type=str,
)
def main(episodes, initial_episode, num_cubes, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(str(assets_path() / "cubes_specs.yml"), "rb") as f:
        cubes_conf = yaml.load(f, Loader=yaml.FullLoader)

    env = gym.make("PRL_UR5-EGL-TowerCamEnv-v0")

    scene = env.unwrapped.scene
    scene.renders(True)

    env = env.unwrapped

    cubes_top = cubes_conf["cubes_top"]
    cubes_bottom = cubes_conf["cubes_bottom"]

    for i in range(episodes):
        top_idx = np.random.choice(len(cubes_top))
        bottom_idx = np.random.choice(len(cubes_bottom))

        cubes_color = [cubes_bottom[bottom_idx]["color"], cubes_top[top_idx]["color"]]
        cubes_size = [cubes_bottom[bottom_idx]["size"], cubes_top[top_idx]["size"]]

        obs = env.reset(cubes_size=cubes_size, cubes_color=cubes_color)

        cubes_size = env.scene._cubes_size
        cubes_position = []
        cubes_color = env.scene._cubes_color

        gripper_position = obs["tool_position"]

        cubes_size = []
        cubes_position = []
        cubes_color = []
        for cube_idx in range(min(num_cubes, len(env.scene._cubes))):
            cubes_position.append(env.scene._cubes[cube_idx].position[0])
            cubes_color.append(env.scene._cubes_color[cube_idx])
            cubes_size.append(env.scene._cubes_size[cube_idx])

        episode_info = {
            "cubes_size": cubes_size,
            "cubes_position": cubes_position,
            "cubes_color": cubes_color,
            "gripper_position": gripper_position,
        }

        with open(str(output_path / f"{i+initial_episode:4d}.pkl"), "wb") as f:
            pkl.dump(episode_info, f)


if __name__ == "__main__":
    main()

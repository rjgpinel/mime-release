import click
import gym
import gym

import matplotlib.pyplot as plt
from mime.agent.utils import Rate
from mime.agent.script_agent import ScriptAgent


@click.command()
@click.option("-e", "--env-name", default="PRL_UR5-PickEasyRandCamEnv-v0", type=str)
@click.option("-s", "--seed", default=0, type=int)
def view(env_name, seed):
    env = gym.make(env_name)
    scene = env.unwrapped.scene
    scene.renders(True)
    rate = Rate(scene.dt)
    env.seed(seed)

    obs = env.reset()
    done = False
    agent = ScriptAgent(env)
    while not done:
        action = agent.get_action()
        obs, reward, done, info = env.step(action)
        rate.sleep()
    # for i in range(5):
    #     plt.imshow(obs[f"rgb{i}"])
    #     plt.show()
    #     plt.imshow(obs[f"depth{i}"])
    #     plt.show()
    #     plt.imshow(obs[f"mask{i}"])
    #     plt.show()


if __name__ == "__main__":
    view()

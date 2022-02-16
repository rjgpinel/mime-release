import numpy as np

from math import pi

from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from .utils import sample_without_overlap


class TowerScene(TableScene):
    def __init__(self, **kwargs):
        super(TowerScene, self).__init__(**kwargs)
        self._modder = TableModder(self)

        self._count_success = 0
        self._num_cubes = 2
        self._cubes = []
        self._cubes_size = []

        self._cubes_size_range = [
            {"low": 0.045, "high": 0.055},
            {"low": 0.04, "high": 0.05},
        ]

        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

    def load(self, np_random):
        super(TowerScene, self).load(np_random)

    def reset(self, np_random, **kwargs):
        super(TowerScene, self).reset(np_random)
        cubes_size = kwargs.get("cubes_size", None)
        cubes_position = kwargs.get("cubes_position", None)
        cubes_color = kwargs.get("cubes_color", None)
        gripper_position = kwargs.get("gripper_position", None)

        modder = self._modder

        # load and randomize cage
        modder.load_cage(np_random)
        if self._domain_rand:
            modder.randomize_cage_visual(np_random)

        self._count_success = 0

        low, high = self._object_workspace
        low_cubes, high_cubes = np.array(low.copy()), np.array(high.copy())

        for cube in self._cubes:
            cube.remove()

        self._cubes = []
        self._cubes_size = []

        if gripper_position is None:
            gripper_pos, gripper_orn = self.random_gripper_pose(np_random)
        else:
            gripper_pos = gripper_position
            gripper_orn = [pi, 0, pi / 2]

        q0 = self.robot.arm.controller.joints_target
        q = self.robot.arm.kinematics.inverse(gripper_pos, gripper_orn, q0)
        self.robot.arm.reset(q)

        # load cubes
        for i in range(self._num_cubes):
            if cubes_size is not None:
                cube_size = cubes_size[i]
                cube, cube_size = modder.load_mesh("cube", cube_size, np_random)
            else:
                cube_size_range = self._cubes_size_range[i]
                cube, cube_size = modder.load_mesh("cube", cube_size_range, np_random)
            self._cubes.append(cube)
            self._cubes_size.append(cube_size)

        # always stack the cubes in the same order
        # use color information to deduce the order
        self._cubes_size = np.array(self._cubes_size)
        low_cubes[:2] += self._cubes_size[0]
        high_cubes[:2] -= self._cubes_size[0]

        rand_color = True
        if cubes_color:
            rand_color = False
            cubes_color = np.array(cubes_color, dtype=np.float)
        else:
            cubes_color = [[218, 86, 80, 255], [128, 196, 99, 255]]
            cubes_color = np.array(cubes_color, dtype=np.float) / 255

        aabbs = []
        self._cubes_color = cubes_color
        # move cubes to a random position and change color
        for i in range(len(self._cubes)):
            cube = self._cubes[i]
            color = cubes_color[i]
            if rand_color:
                # BE CAREFUL! Hardcoded Domain Rand color
                modder.randomize_object_color(np_random, cube, color)
                self._cubes_color[i] = cube.color
            else:
                cube.color = color

            if cubes_position is None:
                aabbs, _ = sample_without_overlap(
                    cube, aabbs, np_random, low_cubes, high_cubes, 0, 0, min_dist=0.04
                )
            else:
                cube.position = cubes_position[i]

    def script(self):
        arm = self.robot.arm
        grip = self.robot.gripper
        cubes_pos = self.cubes_position
        cubes_size = self._cubes_size
        tower_pos = cubes_pos[0]
        tower_height = 0

        sc = Script(self)
        moves = []
        z_offset = np.array([0.0, 0.0, 0.02])
        for pick_pos, placed_cube_size in zip(cubes_pos[1:], cubes_size[:-1]):
            tower_height += placed_cube_size
            pick_cube_size = pick_pos[-1] / 2
            place_height = tower_height + pick_cube_size
            moves += [
                sc.tool_move(arm, pick_pos + [0, 0, 0.1]),
                sc.tool_move(arm, pick_pos + z_offset),
                sc.grip_close(grip),
                sc.tool_move(arm, pick_pos + [0, 0, place_height] + z_offset),
                sc.tool_move(arm, tower_pos + [0, 0, place_height] + z_offset),
                sc.grip_open(grip),
                sc.tool_move(arm, tower_pos + [0, 0, place_height]),
            ]

        return moves

    @property
    def cubes_position(self):
        return np.array([cube.position[0] for cube in self._cubes])

    def distance_to_cubes(self, idx):
        tool_pos, _ = self.robot.arm.tool.state.position
        return np.array(
            [np.subtract(cube.position[0], tool_pos) for cube in self._cubes]
        )

    def get_reward(self, action):
        return 0

    def is_task_success(self):
        cubes_pos = self.cubes_position
        cubes_size = self._cubes_size
        tower_pos = cubes_pos[0]
        heights = np.cumsum(cubes_size)[1:]
        bary_cubes = np.mean(cubes_pos[1:], axis=0)
        cubes_on_tower = (
            np.linalg.norm(np.subtract(tower_pos[:2], bary_cubes[:2])) < 0.04
        )

        heights_ok = True
        for cube_pos, cube_size, height in zip(cubes_pos[1:], cubes_size[1:], heights):
            heights_ok = (
                heights_ok and np.abs(cube_pos[2] + cube_size / 2 - height) < 0.001
            )

        if heights_ok and cubes_on_tower:
            self._count_success += 1
        success = self._count_success > 5
        return success


def test_scene():
    from time import sleep

    scene = TowerScene()
    scene.renders(True)
    np_random = np.random.RandomState(1)
    while True:
        scene.reset(np_random)
        sleep(1)


if __name__ == "__main__":
    test_scene()

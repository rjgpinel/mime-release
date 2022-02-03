import numpy as np

from .table_scene import TableScene
from .table_modder import TableModder
from ..script import Script
from .utils import sample_without_overlap


class TowerScene(TableScene):
    def __init__(self, **kwargs):
        super(TowerScene, self).__init__(**kwargs)
        self._modder = TableModder(self)

        self._count_success = 0
        self._cubes = []
        self._cubes_size = []

        self._cubes_size_range = [
            {"low": 0.03, "high": 0.055},
            {"low": 0.06, "high": 0.08},
        ]

        v, w = self._max_tool_velocity
        self._max_tool_velocity = (1.5 * v, w)

    def load(self, np_random):
        super(TowerScene, self).load(np_random)

    def reset(self, np_random):
        super(TowerScene, self).reset(np_random)
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

        gripper_pos, gripper_orn = self.random_gripper_pose(np_random)
        q0 = self.robot.arm.controller.joints_target
        q = self.robot.arm.kinematics.inverse(gripper_pos, gripper_orn, q0)
        self.robot.arm.reset(q)

        # load cubes
        cubes = []
        cubes_size = []
        for i in range(len(self._cubes_size_range)):
            cube_size_range = self._cubes_size_range[i]
            cube, cube_size = modder.load_mesh("cube", cube_size_range, np_random)
            cubes.append(cube)
            cubes_size.append(cube_size)

        # sort cubes per decreasing size
        # biggest cube first
        # idxs_sort = np.argsort(-np.array(cubes_size))
        # for idx in idxs_sort:
        #     self._cubes.append(cubes[idx])
        #     self._cubes_size.append(cubes_size[idx])
        self._cubes = cubes
        self._cubes_size = np.array(cubes_size)

        low_cubes[:2] += self._cubes_size[0] / 2
        high_cubes[:2] -= self._cubes_size[0] / 2

        # move cubes to a random position and change color
        aabbs = []
        colors = np.array(
            [
                [11, 124, 96, 255],
                [255, 140, 20, 255],
            ],
            dtype=float,
        )
        colors = colors / 255
        for cube, color in zip(self._cubes, colors):
            aabbs, _ = sample_without_overlap(
                cube, aabbs, np_random, low_cubes, high_cubes, 0, 0, min_dist=0.04
            )
            cube.color = color
            # if self._domain_rand:
            # modder.randomize_object_color(np_random, cube, color)

    def script(self):
        arm = self.robot.arm
        grip = self.robot.gripper
        cubes_pos = self.cubes_position
        cubes_size = self._cubes_size
        tower_pos = cubes_pos[0]
        height = 0

        sc = Script(self)
        moves = []
        z_offset = 0.02
        for pick_pos, cube_size in zip(cubes_pos[1:], cubes_size[:-1]):
            height += cube_size
            moves += [
                sc.tool_move(arm, pick_pos + [0, 0, 0.1]),
                sc.tool_move(arm, pick_pos + [0, 0, z_offset]),
                sc.grip_close(grip),
                sc.tool_move(arm, pick_pos + [0, 0, height + z_offset * 2]),
                sc.tool_move(arm, tower_pos + [0, 0, height + z_offset * 2]),
                sc.grip_open(grip),
                sc.tool_move(arm, tower_pos + [0, 0, height + cube_size]),
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

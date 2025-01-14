import collections

import numpy as np
import pybullet as pb

from . import collision
from .rope import Rope
from .dynamics import Dynamics
from .joint import Joint, JointArray
from .link import Link
from .shape import VisualShape, CollisionShape
from .utils import augment_path


class Body(JointArray):
    def __init__(self, body_id, client_id, egl=False):
        self.client_id = client_id
        self.egl = egl
        num_joints = pb.getNumJoints(body_id, physicsClientId=client_id)
        super(Body, self).__init__(body_id, list(range(num_joints)), client_id)

    @staticmethod
    def num_bodies(client_id):
        return pb.getNumBodies(physicsClientId=client_id)

    @property
    def name(self):
        info = pb.getBodyInfo(self.body_id, physicsClientId=self.client_id)
        return info[-1].decode("utf8")

    def link(self, name):
        return next(
            (
                Link(self.body_id, i.index, self.client_id)
                for i in self.info
                if i.link_name == name
            ),
            None,
        )

    def joint(self, name):
        return next(
            (
                Joint(self.body_id, i.index, self.client_id)
                for i in self.info
                if i.joint_name == name
            ),
            None,
        )

    def joints(self, names):
        indices = []
        for name in names:
            ind = next((i.index for i in self.info if i.joint_name == name), None)
            assert ind is not None, 'Unknown joint "{}" on body {} "{}"'.format(
                name, self.body_id, self.name
            )
            indices.append(ind)
        return JointArray(self.body_id, indices, self.client_id)

    def links(self):
        return (Link(self.body_id, i, self.client_id) for i in self.joint_indices)

    def __getitem__(self, key):
        if isinstance(key, collections.Iterable):
            return JointArray(self._body_id, key, self.client_id)
        else:
            return super(Body, self).__getitem__(key)

    @property
    def position(self):
        return pb.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.client_id
        )

    @position.setter
    def position(self, pos_orn):
        if len(pos_orn) == 2:
            pos, orn = pos_orn
        else:
            pos, orn = pos_orn, (0, 0, 0, 1)
        if len(orn) == 3:
            orn = pb.getQuaternionFromEuler(orn)
        pb.resetBasePositionAndOrientation(
            self.body_id, pos, orn, physicsClientId=self.client_id
        )

    @property
    def color(self):
        return self.visual_shape.rgba_color

    @color.setter
    def color(self, value):
        pb.changeVisualShape(
            self.body_id,
            -1,
            rgbaColor=value,
        )

    def randomize_shape_link(self, np_random, textures, link_id):
        if len(textures) > 0:
            texture_id = np_random.choice(textures)
        else:
            texture_id = -1
        color = np.append(np_random.uniform(0, 1, 3), 1)
        pb.changeVisualShape(
            self.body_id,
            link_id,
            textureUniqueId=texture_id,
            rgbaColor=color,
        )

    def randomize_shape(self, np_random, textures, per_link=True):
        self.randomize_shape_link(np_random, textures, -1)
        if per_link:
            for link_id in range(pb.getNumJoints(self.body_id)):
                self.randomize_shape_link(np_random, textures, link_id)

    @property
    def visual_shape(self):
        return VisualShape(self._body_id, -1, self.client_id)

    @property
    def collision_shape(self):
        return CollisionShape(self._body_id, -1, self.client_id)

    @property
    def dynamics(self):
        return Dynamics(self._body_id, -1, self.client_id)

    def get_overlapping_objects(self):
        """Return all the unique ids of objects that have axis aligned
        bounding box overlap with a axis aligned bounding box of
        a given body."""
        return collision.get_overlapping_objects(self)

    def get_contacts(self, body_or_link_b=None):
        """Returns the contact points computed during the most recent
        call to stepSimulation."""
        return collision.get_contact_points(self, body_or_link_b)

    def get_closest_points(self, max_distance, body_or_link_b=None):
        """Compute the closest points, independent from stepSimulation.
        If the distance between objects exceeds this maximum distance,
        no points may be returned."""
        return collision.get_closest_points(self, body_or_link_b, max_distance)

    def get_collisions(self):
        """Return all objects that intersect a given body."""
        return collision.get_collisions(self)

    def remove(self):
        pb.removeBody(self.body_id, physicsClientId=self.client_id)

    @staticmethod
    def load(file_name, client_id, egl=False, **kwargs):
        path = augment_path(file_name)
        loader = {".urdf": pb.loadURDF, ".xml": pb.loadMJCF, ".sdf": pb.loadSDF}
        loader = loader[path.suffix.lower()]
        ids = loader(str(path), physicsClientId=client_id, **kwargs)
        if isinstance(ids, collections.Iterable):
            return [Body(i, client_id, egl) for i in ids]
        return Body(ids, client_id, egl)

    @staticmethod
    def create(
        visual_id,
        collision_id,
        client_id,
        pos=(0, 0, 0),
        orn=(0, 0, 0, 1),
        mass=0,
        egl=False,
    ):
        body_id = pb.createMultiBody(
            baseVisualShapeIndex=visual_id,
            baseCollisionShapeIndex=collision_id,
            basePosition=pos,
            baseOrientation=orn,
            baseMass=mass,
            physicsClientId=client_id,
        )
        return Body(body_id, client_id, egl)

    @staticmethod
    def box(size, client_id, collision=False, mass=0, egl=False):
        size = np.array(size) / 2
        vis_id = pb.createVisualShape(
            pb.GEOM_BOX, halfExtents=size, physicsClientId=client_id
        )
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=size, physicsClientId=client_id
            )
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def sphere(radius, client_id, collision=False, mass=0, egl=False):
        vis_id = pb.createVisualShape(
            pb.GEOM_SPHERE, radius=radius, physicsClientId=client_id
        )
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_SPHERE, radius=radius, physicsClientId=client_id
            )
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def cylinder(radius, height, client_id, collision=False, mass=0, egl=False):
        vis_id = pb.createVisualShape(
            pb.GEOM_CYLINDER, radius=radius, length=height, physicsClientId=client_id
        )
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_CYLINDER,
                radius=radius,
                height=height,
                physicsClientId=client_id,
            )
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def capsule(radius, height, client_id, collision=False, mass=0, egl=False):
        vis_id = pb.createVisualShape(
            pb.GEOM_CAPSULE, radius=radius, length=height, physicsClientId=client_id
        )
        col_id = -1
        if collision:
            col_id = pb.createCollisionShape(
                pb.GEOM_CAPSULE, radius=radius, height=height, physicsClientId=client_id
            )
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    @staticmethod
    def rope(
        position,
        length,
        client_id,
        np_random,
        n_parts=20,
        radius=0.005,
        color=[1, 1, 1],
        color_ends=[1, 1, 1],
        shape_key="xcurly",
        egl=False,
    ):
        position = np.float32(position)
        increment = length / n_parts

        # Parts Visual / Collision
        part_shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[radius] * 3)
        part_visual = pb.createVisualShape(pb.GEOM_SPHERE, radius=radius * 1.5)
        # part_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[radius] * 3)

        XCURLY_SHAPE = [
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 0, 0]),
            np.array([1, -1, 0]),
            np.array([1, -1, 0]),
            np.array([1, -1, 0]),
            np.array([1, -1, 0]),
            np.array([1, -1, 0]),
            np.array([1, -1, 0]),
        ]

        YCURLY_SHAPE = [
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 1, 0]),
            np.array([1, 0, 0]),
            np.array([-1, 1, 0]),
            np.array([-1, 1, 0]),
            np.array([-1, 1, 0]),
            np.array([-1, 1, 0]),
        ]

        CIRCLE_SHAPE = [
            np.array([1, -1, 0]),
            np.array([1, -1, 0]),
            np.array([0, -1, 0]),
            np.array([0, -1, 0]),
            np.array([-1, -1, 0]),
            np.array([-1, -1, 0]),
            np.array([-1, 0, 0]),
            np.array([-1, 0, 0]),
            np.array([-1, 1, 0]),
            np.array([-1, 1, 0]),
            np.array([0, 1, 0]),
            np.array([0, 1, 0]),
        ]

        RANDOM_SHAPE = [
            np.array([0, 0, 1]),
        ]

        shapes = {
            "circle": CIRCLE_SHAPE,
            "xcurly": XCURLY_SHAPE,
            "ycurly": YCURLY_SHAPE,
            "random": RANDOM_SHAPE,
        }

        parent_id = -1
        bodies = []
        for i in range(n_parts):
            if i in [0, 1, n_parts - 2, n_parts - 1]:
                increment_idx = np.array([1, 0, 0])
            else:
                increment_arr = shapes[shape_key]
                if "xcurly":
                    increment_arr_idx = np_random.choice(
                        list(range(len(increment_arr)))
                    )
                    increment_idx = increment_arr[increment_arr_idx]
                else:
                    increment_idx = increment_arr[i % len(increment_arr)]
            increment_v = increment / np.sqrt(np.sum(np.abs(increment_idx)))
            increment_arr = increment_idx * increment_v
            position += increment_arr

            # Base mass = 0.1
            part_id = pb.createMultiBody(
                0.1, part_shape, part_visual, basePosition=position
            )
            if parent_id > -1:
                constraint_id = pb.createConstraint(
                    parentBodyUniqueId=parent_id,
                    parentLinkIndex=-1,
                    childBodyUniqueId=part_id,
                    childLinkIndex=-1,
                    jointType=pb.JOINT_POINT2POINT,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=increment_arr,
                    childFramePosition=(0, 0, 0),
                )
                pb.changeConstraint(constraint_id, maxForce=100)
            if (i > 0) and (i < n_parts - 1):
                part_color = color + [1]
            else:
                part_color = color_ends + [1]
            pb.changeVisualShape(part_id, -1, rgbaColor=part_color)
            parent_id = part_id
            bodies.append(Body(part_id, client_id, egl))
        rope = Rope(bodies, length, increment, radius)
        return rope

    @staticmethod
    def mesh(
        file_name, client_id, collision_file_name=None, scale=1, mass=0, egl=False
    ):
        path = str(augment_path(file_name))
        if np.isscalar(scale):
            scale = (scale,) * 3
        vis_id = pb.createVisualShape(
            pb.GEOM_MESH, fileName=path, meshScale=scale, physicsClientId=client_id
        )
        col_id = -1
        if collision_file_name is not None:
            path = str(augment_path(collision_file_name))
            col_id = pb.createCollisionShape(
                pb.GEOM_MESH, fileName=path, meshScale=scale, physicsClientId=client_id
            )
        return Body.create(vis_id, col_id, client_id, mass=mass, egl=egl)

    def __eq__(self, other):
        return self._body_id == other.body_id

    def __hash__(self):
        return self._body_id

    def __repr__(self):
        return 'Body({}) ""'.format(self._body_id, self.name)

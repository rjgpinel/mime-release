#!/usr/bin/env python3
from mime.scene import Body
from mime.settings import SHAPENET_PATH


class MeshLoader:
    def __init__(
        self,
        client_id,
        folders,
        ratios=(1,),
        train=True,
        simplified=False,
        verbose=False,
        egl=False,
        reduced_radius_model=False,
    ):
        self.client_id = client_id
        self.simplified = simplified
        self.reduced_radius_model = reduced_radius_model
        self.root = SHAPENET_PATH
        self.egl = egl
        self.folders = folders
        self.files = []
        self.dic_labels = json.load(open(os.path.join(self.root, "labels.json"), "r"))

        for folder, ratio in zip(folders, ratios):
            self.files_folder = [
                os.path.join(folder, file)
                for file in self.dic_labels[folder]["train"]
                + self.dic_labels[folder]["test"]
            ]
            self.files_folder = self.files_folder[: int(ratio * len(self.files_folder))]
            self.files += self.files_folder

        self.verbose = verbose
        if self.verbose:
            print("Folder {} Num {}".format(folder, len(self.files)))

    def __len__(self):
        return len(self.files)

    def get_mesh(self, idx, scale, reduced_radius_model=False, useFixedBase=False):
        if self.simplified and os.path.exists(
            os.path.join(self.root, self.files[idx], "models", "model_simplified.urdf")
        ):
            if self.reduced_radius_model and os.path.exists(
                os.path.join(
                    self.root, self.files[idx], "models", "model_simplified_radius.urdf"
                )
            ):
                urdf = "model_simplified_radius.urdf"
            else:
                urdf = "model_simplified.urdf"
        else:
            urdf = "model_normalized.urdf"
        path_mesh = os.path.join(self.root, self.files[idx], "models", urdf)

        if self.verbose:
            print("Loading mesh", path_mesh)

        mesh = Body.load(
            path_mesh,
            client_id=self.client_id,
            globalScaling=scale,
            flags=pb.URDF_USE_IMPLICIT_CYLINDER,
            egl=self.egl,
            useFixedBase=useFixedBase,
        )
        return mesh

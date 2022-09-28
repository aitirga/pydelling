from pathlib import Path

import meshio
import numpy as np
import trimesh
import trimesh.proximity as proximity


class Fault:
    local_id = 0
    def __init__(self, filename=None,
                 mesh=None,
                 polygon=None,
                 aperture=None,
                 transmissivity=None,
                 porosity=None,
                 storativity=None,
                 effective_aperture=None,
                 ):

        if filename is not None:
            self.meshio_mesh = meshio.read(filename)
        if mesh is not None:
            self.meshio_mesh: meshio.Mesh = mesh
        self.trimesh_mesh: trimesh.Trimesh = trimesh.load_mesh(filename)
        self.aperture = aperture
        self.associated_elements = []
        self.transmissivity = transmissivity
        self.porosity = porosity
        self.storativity = storativity
        self.filename = filename
        self.local_id = Fault.local_id
        self.effective_aperture=effective_aperture
        Fault.local_id += 1

    def distance(self, points: np.ndarray, n_max: int = 2500):
        if points.shape[0] == 3:
            points = points.reshape(-1, 3)
        # Divide the points into chunks of n_max
        n_chunks = int(points.shape[0] / n_max)
        if n_chunks == 0:
            n_chunks = 1
        distances = []
        for i in range(n_chunks):
            distances.append(proximity.signed_distance(self.trimesh_mesh, points[i * n_max:(i + 1) * n_max]))
        return np.concatenate(distances)

    def _to_obj(self, global_id=0):
        """Converts the fault to an obj file"""
        str_obj = ""
        for i, f in enumerate(self.meshio_mesh.points):
            str_obj += f"v {f[0]} {f[1]} {f[2]}\n"

        for i, f in enumerate(self.meshio_mesh.cells[0].data):
            str_obj += "f "
            for j in range(3):
                str_obj += str(f[j] + global_id) + " "
            str_obj += "\n"
        return str_obj

    def to_obj(self, filename=None, global_id=1):
        str_obj = self._to_obj(global_id=global_id)
        if filename is not None:
            with open(filename, 'w') as f:
                f.write(str_obj)
        return str_obj


    @property
    def points(self):
        return self.meshio_mesh.points

    @property
    def cells(self):
        return self.meshio_mesh.cells[0].data

    @property
    def num_points(self):
        return self.meshio_mesh.points.shape[0]

    @property
    def num_cells(self):
        return self.meshio_mesh.cells[0].data.shape[0]

    @property
    def centroid(self):
        return self.trimesh_mesh.centroid

    @property
    def size(self):
        return np.sqrt(self.trimesh_mesh.area)

    @property
    def normal_vector(self):
        return np.mean(self.trimesh_mesh.face_normals, axis=0)

    def get_json(self):
        self.filename: Path
        save_dict = {
            "aperture": self.aperture,
            "transmissivity": self.transmissivity,
            "porosity": self.porosity,
            "storativity": self.storativity,
            "filename": str(self.filename),
        }
        return save_dict

    @property
    def area(self):
        return self.trimesh_mesh.area

    def __str__(self):
        return f"Fault {self.local_id}"








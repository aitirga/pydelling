import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor
import logging
import vtk
logger = logging.getLogger(__name__)
import meshio as msh
from tqdm import tqdm
import meshio


class VTKMeshReader(MeshPreprocessor):
    has_kd_tree = False

    def __init__(self, filename, kd_tree=True):
        super().__init__()
        self.meshio_mesh: meshio.Mesh = meshio.read(filename)
        self._coords = self.meshio_mesh.points
        self.convert_meshio_to_meshpreprocessor()
        if kd_tree:
            self.create_kd_tree()
            self.has_kd_tree = True

    def convert_meshio_to_meshpreprocessor(self):
        for cell_block in self.meshio_mesh.cells:
            element_type = cell_block.type
            if element_type == "wedge":
                for element in tqdm(cell_block.data, desc='Setting up wedge elements'):
                    self.add_wedge(element, self._coords[element])

            if element_type == 'hexahedron':
                for element in tqdm(cell_block.data, desc='Setting up hexahedron elements'):
                    self.add_hexahedra(element, self._coords[element])

            if element_type == 'tetra':
                for element in tqdm(cell_block.data, desc='Setting up tetra elements'):
                    self.add_tetrahedra(element, self._coords[element])



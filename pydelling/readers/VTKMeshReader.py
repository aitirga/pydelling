import os
import numpy as np
from ..preprocessing.mesh_preprocessor import MeshPreprocessor
import logging
import vtk
logger = logging.getLogger(__name__)
import meshio as msh
from tqdm import tqdm
import meshio
from pathlib import Path


class VTKMeshReader(MeshPreprocessor):
    has_kd_tree = False

    def __init__(self, filename, kd_tree=True):
        super().__init__()
        if Path(filename).suffix == '.vtk':
            self.meshio_mesh: meshio.Mesh = meshio.read(filename)
            self._coords = self.meshio_mesh.points
            self.convert_meshio_to_meshpreprocessor()
            if kd_tree:
                self.create_kd_tree()
                self.has_kd_tree = True
        else:
            self.load(filename)

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

    def save(self, filename):
        """Save the mesh to a file."""
        import pickle
        logger.info(f'Saving mesh to {filename}')
        with open(filename, 'wb') as f:
            save_dictionary = {
                'elements': self.elements,
                'coords': self.coords,
                'kd_tree': self.kd_tree,
                'has_kd_tree': self.has_kd_tree,
            }
            pickle.dump(save_dictionary, f)

    def load(self, filename):
        """Load the mesh from a file."""
        logger.info(f'Loading mesh from {filename}')
        import pickle
        with open(filename, 'rb') as f:
            save_dictionary = pickle.load(f)
            self.elements = save_dictionary['elements']
            self._coords = save_dictionary['coords']
            self.kd_tree = save_dictionary['kd_tree']
            self.has_kd_tree = save_dictionary['has_kd_tree']



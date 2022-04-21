import os
import numpy as np
from ..preprocessing.MeshPreprocessor import MeshPreprocessor
import logging
import vtk
logger = logging.getLogger(__name__)
import meshio as msh
from tqdm import tqdm


class VTKMeshReader(MeshPreprocessor):

    def __init__(self, filename):
        super().__init__()
        self.open_file(filename)

    def open_file(self, filename):

        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        vtk_mesh = reader.GetOutput()

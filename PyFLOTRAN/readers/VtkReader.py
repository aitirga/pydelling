import numpy as np
from .BaseReader import BaseReader
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy


class VtkReader(BaseReader):
    """
    This class provides the framework to read data from a VTK file and do different postprocessing steps
    """
    def read_file(self, opened_file):
        self.vtk_file = LegacyVTKReader(FileNames=self.filename)

    @property
    def values(self):
        _vtk_object = sm.Fetch(self.vtk_file)
        _vtk_object = dsa.WrapDataObject(_vtk_object)
        return _vtk_object.PointData

    @property
    def keys(self):
        _vtk_object = sm.Fetch(self.vtk_file)
        return self.vtk_file.PointData.keys()
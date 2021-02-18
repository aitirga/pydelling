import numpy as np
from .BaseReader import BaseReader
try:
    from paraview.simple import *
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview import servermanager as sm
except:
    pass
from vtk.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData


class XDMFReader(BaseReader):
    """
    This class provides the framework to read data from a XDMF file
    """
    current_array: None
    calculator: None

    def read_file(self, opened_file):
        # self.vtk_file = LegacyVTKReader(FileNames=self.filename)
        # self.current_array = self.vtk_file
        """
        Reads the data and stores it inside the class
        :return:
        """
        pass
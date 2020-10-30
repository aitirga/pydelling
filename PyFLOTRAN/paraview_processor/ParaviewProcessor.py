import numpy as np
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview import servermanager as sm
from typing import Dict


class ParaviewProcessor:
    """
    This class provides the framework to read data from a VTK file and do different postprocessing steps
    """
    current_array: None
    calculator: Calculator
    pipeline: Dict[str, object]
    read_data_counter: int = 0

    def __init__(self):
        pass

    def read_vtk_file(self):
        print(self.read_data_counter)
        self.vtk_file = LegacyVTKReader(FileNames=self.filename)
        self.current_array = self.vtk_file

    @property
    def values(self):
        _vtk_object = sm.Fetch(self.current_array)
        _vtk_object = dsa.WrapDataObject(_vtk_object)
        return _vtk_object.PointData

    @property
    def keys(self):
        return self.current_array

    @property
    def data_keys(self):
        _vtk_object = sm.Fetch(self.vtk_file)
        return self.vtk_file.PointData.keys()

    @property
    def data_values(self):
        _vtk_object = sm.Fetch(self.vtk_file)
        _vtk_object = dsa.WrapDataObject(_vtk_object)
        return _vtk_object.PointData

    def add_calculator(self, input=None, function=''):
        """
        Adds a calculator filter to a dataset
        Returns:
            The Calculator object
        """
        input = input if input else self.current_array
        self.calculator = Calculator(Input=input)
        self.calculator.Function = function

        self.current_array = self.calculator
        return self.current_array
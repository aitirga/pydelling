from PyFLOTRAN.paraview_processor.filters import BaseFilter
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy


class IntegrateVariablesFilter(BaseFilter):
    """
    This class implements the Paraview Integrate Variables filter
    """
    filter_type: str = "Integrate_variables"
    counter: int = 0

    def __init__(self, input_filter, name, divide_cell_data_by_volume=False):
        super().__init__(name=name)
        IntegrateVariablesFilter.counter += 1
        self.filter = IntegrateVariables(Input=input_filter)
        self.set_divide_cell_data_by_volume(divide_cell_data_by_volume)

    def set_divide_cell_data_by_volume(self, value):
        self.filter.DivideCellDataByVolume = value


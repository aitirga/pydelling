import pandas as pd
import numpy as np
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy

class BaseFilter:
    filter_type: str = "VTK_reader"
    counter: int = 0
    filter: object

    def __init__(self, name):
        self.name = name

    @property
    def cell_keys(self):
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        return vtk_object.CellData.keys()

    @property
    def point_keys(self):
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        return vtk_object.PointData.keys()

    @property
    def cell_data(self):
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        pd_df = pd.DataFrame([var for var in vtk_object.CellData])
        pd_df = pd_df.transpose()
        pd_df.columns = self.cell_keys
        return pd_df

    @property
    def point_data(self):
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        pd_df = pd.DataFrame([var for var in vtk_object.PointData])
        pd_df = pd_df.transpose()
        pd_df.columns = self.point_keys
        return pd_df
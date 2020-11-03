import pandas as pd
import numpy as np
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview.vtk.numpy_interface import algorithms as algs
from paraview import servermanager as sm
from vtk.util.numpy_support import vtk_to_numpy
from typing import List


class BaseFilter:
    filter_type: str = "VTK_reader"
    counter: int = 0
    filter: object
    vector_keys: List = ["x", "y", "z"]

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
    def field_keys(self):
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        return vtk_object.FieldData.keys()

    @property
    def cell_data(self) -> pd.DataFrame:
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        pd_df = pd.DataFrame()
        for key in self.cell_keys:
            temp_dataset = np.array(vtk_object.CellData[key]).transpose()
            if len(temp_dataset.shape) != 1:
                # The dataset is a vector:
                for idx, vector_element in enumerate(temp_dataset):
                    new_key = f"{key}{self.vector_keys[idx]}"
                    pd_df[new_key] = vector_element
            else:
                pd_df[key] = temp_dataset
        return pd_df

    @property
    def point_data(self) -> pd.DataFrame:
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        pd_df = pd.DataFrame()
        for key in self.point_keys:
            temp_dataset = np.array(vtk_object.PointData[key]).transpose()
            if len(temp_dataset.shape) != 1:
                # The dataset is a vector:
                for idx, vector_element in enumerate(temp_dataset):
                    new_key = f"{key}{self.vector_keys[idx]}"
                    pd_df[new_key] = vector_element
            else:
                pd_df[key] = temp_dataset
        return pd_df


    @property
    def field_data(self) -> pd.DataFrame:
        vtk_object = sm.Fetch(self.filter)
        vtk_object = dsa.WrapDataObject(vtk_object)
        pd_df = pd.DataFrame()
        for key in self.field_keys:
            temp_dataset = np.array(vtk_object.FieldData[key]).transpose()
            if len(temp_dataset.shape) != 1:
                # The dataset is a vector:
                for idx, vector_element in enumerate(temp_dataset):
                    new_key = f"{key}{self.vector_keys[idx]}"
                    pd_df[new_key] = vector_element
            else:
                pd_df[key] = temp_dataset
        return pd_df

    @property
    def mesh_points(self) -> pd.DataFrame:
        vtk_object = sm.Fetch(self.filter)
        return pd.DataFrame(vtk_to_numpy(vtk_object.GetPoints().GetData()), columns=["x", "y", "z"])
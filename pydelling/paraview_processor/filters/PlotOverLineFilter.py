from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface import algorithms as algs
    from paraview import servermanager as sm
except:
    pass


import pandas as pd
import numpy as np


class PlotOverLineFilter(BaseFilter):
    """
    This class implements the PlotOverLine paraview filter
    """
    filter_type: str = "PlotOverLineFilter"
    counter: int = 0

    def __init__(self, input_filter, name, point_1=None, point_2=None, n_line=None):
        super().__init__(name=name)
        PlotOverLineFilter.counter += 1
        self.filter = PlotOverLine(Input=input_filter)
        if point_1:
            assert point_2, "Two points need to be defined"
        if point_2:
            assert point_1, "Two points need to be defined"
        if point_1 or point_2:
            self.filter.Point1 = point_1
            self.filter.Point2 = point_2
        if n_line:
            self.filter.Resolution = n_line

    def set_points(self, point_1, point_2):
        """
        This method sets the `point_1` and `point_2` variables to the filter
        Args:
            point_1: First point of the line
            point_2: Second point of the line
        """
        self.filter.Point1 = point_1
        self.filter.Point2 = point_2

    def set_line_resolution(self, n):
        """
        This methods modifies the resolution of the line used to interpolate the data
        Args:
            n: Number specifying the number of divisions of the line used to interpolate the data on.
        """
        self.filter.Resolution = n

    @property
    def point_data(self):
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
        return pd_df.dropna()



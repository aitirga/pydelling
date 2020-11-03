from PyFLOTRAN.paraview_processor.filters import BaseFilter
from paraview.simple import *


class PlotOverLineFilter(BaseFilter):
    """
    This class implements the PlotOverLine paraview filter
    """
    filter_type: str = "PlotOverLineFilter"
    counter: int = 0

    def __init__(self, input_filter, name, point_1=None, point_2=None):
        super().__init__(name=name)
        PlotOverLineFilter.counter += 1
        self.filter = PlotOverLine(Input=input_filter)
        if point_1:
            assert point_2, "Two points need to be defined"
        if point_2:
            assert point_1, "Two points need to be defined"
        if point_1 or point_2:
            self.filter.Source.Point1 = point_1
            self.filter.Source.Point2 = point_2

    def set_points(self, point_1, point_2):
        """
        This method sets the `point_1` and `point_2` variables to the filter
        Args:
            point_1: First point of the line
            point_2: Second point of the line
        """
        self.filter.Source.Point1 = point_1
        self.filter.Source.Point2 = point_2


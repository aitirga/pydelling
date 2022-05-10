from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class CellDataToPointDataFilter(BaseFilter):
    """
    This class implements the Paraview Integrate Variables filter
    """
    filter_type: str = "Cell_data_to_point_data"
    counter: int = 0

    def __init__(self, input_filter, name):
        super().__init__(name=name)
        CellDataToPointDataFilter.counter += 1
        self.filter = CellDatatoPointData(Input=input_filter)



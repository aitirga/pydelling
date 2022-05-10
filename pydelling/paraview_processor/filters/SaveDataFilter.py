from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class SaveDataFilter(BaseFilter):
    """
    This class implements the Save Data filter
    """
    filter_type: str = "Save_data"
    counter: int = 0

    def __init__(self, filename, proxy, point_data_arrays, cell_data_arrays, name):
        super().__init__(name=name)
        SaveDataFilter.counter += 1
        self.filter = SaveData(self, Input=filename, proxy=proxy)
        self.filter.Proxy = proxy
        self.filter.PointDataArrays = point_data_arrays
        self.filter.CellDataArrays = cell_data_arrays

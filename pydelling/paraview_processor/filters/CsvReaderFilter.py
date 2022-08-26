from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass
from functools import wraps


def convert_to_table(func):
    @wraps(func)
    def wrapper():

        func()
    return wrapper


class CsvReaderFilter(BaseFilter):
    filter_type: str = "VTK_reader"
    counter: int = 0
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __init__(self, filename, name, coordinate_labels=("x", "y", "z")):
        super().__init__(name=name)
        self.filter = CSVReader(FileName=str(filename))
        CsvReaderFilter.counter += 1
        self.coordinate_labels = coordinate_labels
        self.filter = self.convert_table_to_points()
        self.set_ranges()

    def set_ranges(self):
        self.x_min = self.mesh_points.min()["x"]
        self.x_max = self.mesh_points.max()["x"]
        self.y_min = self.mesh_points.min()["y"]
        self.y_max = self.mesh_points.max()["y"]
        self.z_min = self.mesh_points.min()["z"]
        self.z_max = self.mesh_points.max()["z"]

    def convert_table_to_points(self):
        _table_to_points = TableToPoints(Input=self.filter)
        _table_to_points.XColumn = self.coordinate_labels[0]
        _table_to_points.YColumn = self.coordinate_labels[1]
        _table_to_points.ZColumn = self.coordinate_labels[2]
        return _table_to_points
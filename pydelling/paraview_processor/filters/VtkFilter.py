from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class VtkFilter(BaseFilter):
    filter_type: str = "VTK_reader"
    counter: int = 0
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __init__(self, filename, name):
        super().__init__(name=name)
        self.filter = LegacyVTKReader(FileNames=str(filename))
        VtkFilter.counter += 1
        self.set_ranges()

    def set_ranges(self):
        self.x_min = self.mesh_points.min()["x"]
        self.x_max = self.mesh_points.max()["x"]
        self.y_min = self.mesh_points.min()["y"]
        self.y_max = self.mesh_points.max()["y"]
        self.z_min = self.mesh_points.min()["z"]
        self.z_max = self.mesh_points.max()["z"]
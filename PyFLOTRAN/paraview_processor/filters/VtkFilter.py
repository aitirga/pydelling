from PyFLOTRAN.paraview_processor.filters import BaseFilter
from paraview.simple import *


class VtkFilter(BaseFilter):
    filter_type: str = "VTK_reader"
    counter: int = 0

    def __init__(self, filename, name):
        super().__init__(name=name)
        self.filter = LegacyVTKReader(FileNames=str(filename))
        VtkFilter.counter += 1
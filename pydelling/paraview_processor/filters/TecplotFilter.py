from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class TecplotFilter(BaseFilter):
    filter_type: str = "tecplot_reader"
    counter: int = 0

    def __init__(self, filename, name):
        super().__init__(name=name)
        self.filter = TecplotReader(FileNames=str(filename))
        TecplotFilter.counter += 1


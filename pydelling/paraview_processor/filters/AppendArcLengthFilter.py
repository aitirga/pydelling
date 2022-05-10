from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class AppendArcLengthFilter(BaseFilter):
    """
    This class implements the Append Arc-Length filter
    """
    filter_type: str = "Append_arc_length"
    counter: int = 0

    def __init__(self, input_filter, name):
        super().__init__(name=name)
        AppendArcLengthFilter.counter += 1
        self.filter = AppendArcLength(Input=input_filter)
from PyFLOTRAN.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class ClipFilter(BaseFilter):
    """
    This class implements the Paraview Integrate Variables filter
    """
    filter_type: str = "Clip"
    counter: int = 0

    def __init__(self, input_filter, name, *args, **kwargs):
        super().__init__(name=name)
        ClipFilter.counter += 1
        self.filter = Clip(Input=input_filter)
        for kwarg in kwargs:
            setattr(self.filter, kwarg, kwargs[kwarg])

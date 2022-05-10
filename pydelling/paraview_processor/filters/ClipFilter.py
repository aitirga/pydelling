from pydelling.paraview_processor.filters import BaseFilter
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

    def clip_box(self, box_position, box_length, exact = True):
        """
        This method clips a box like domain into the filter
        Args:
            box_position: bottom left corner coordinates of the box.
            box_length: length vector of the box.
            exact: if true, the box will clip exactly the source filter.
        """
        self.filter.ClipType = 'Box'
        self.filter.ClipType.Position = box_position
        self.filter.ClipType.Length = box_length
        self.filter.Exact = exact

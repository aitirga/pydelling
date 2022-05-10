from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass




class IntegrateVariablesFilter(BaseFilter):
    """
    This class implements the Paraview Integrate Variables filter
    """
    filter_type: str = "Integrate_variables"
    counter: int = 0

    def __init__(self, input_filter, name, divide_cell_data_by_volume=False):
        super().__init__(name=name)
        IntegrateVariablesFilter.counter += 1
        self.filter = IntegrateVariables(Input=input_filter)
        self.set_divide_cell_data_by_volume(divide_cell_data_by_volume)

    def set_divide_cell_data_by_volume(self, value):
        self.filter.DivideCellDataByVolume = value


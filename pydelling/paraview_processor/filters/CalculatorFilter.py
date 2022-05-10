from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class CalculatorFilter(BaseFilter):
    filter_type: str = "Calculator"
    counter: int = 0

    def __init__(self, input_filter, function, name, output_array_name=None, attribute_type='Cell Data'):
        super().__init__(name=name)
        CalculatorFilter.counter += 1
        self.filter = Calculator(Input=input_filter)
        self.filter.Function = function
        self.filter.AttributeType = attribute_type
        if output_array_name:
            self.filter.ResultArrayName = output_array_name

    def set_attribute_type(self, attribute_type):
        self.filter.AttributeType = attribute_type

    def set_function(self, function):
        self.filter.Function = function

    @property
    def calculation(self):
        """
        This property returns the calculated array
        Returns:
            The array containing the calculation
        """
        if self.filter.AttributeType == "Cell Data":
            return self.cell_data[self.filter.ResultArrayName]
        if self.filter.AttributeType == "Point Data":
            return self.point_data[self.filter.ResultArrayName]


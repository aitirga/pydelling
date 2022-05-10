from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class StreamTracerWithCustomSourceFilter(BaseFilter):
    """
    This class implements the Stream Tracer with custom source filter
    """
    filter_type: str = "Stream_tracer_with_custom_source"
    counter: int = 0

    def __init__(self, input_filter, seed_source, name):
        super().__init__(name=name)
        StreamTracerWithCustomSourceFilter.counter += 1
        self.filter = StreamTracerWithCustomSource(Input=input_filter)
        self.filter.SeedSource = seed_source

    def set_seed_source(self, seed_source):
        self.filter.SeedSource = seed_source

    def add_vector(self, vector_name):
        self.filter.Vectors = ['POINTS', vector_name]

    def add_minimum_step_length(self, minimum_step_length):
        self.filter.MinimumStepLength = minimum_step_length
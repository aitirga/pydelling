from pydelling.readers import RasterFileReader


class ClosedStlGenerator(object):
    """This class generates a closed STL from two regular raster files"""
    def __init__(self, bottom_surface: RasterFileReader,
                 top_surface: RasterFileReader = None,
                 elevation: RasterFileReader = None,
    ):
        # Generate needed data
        self.bottom_surface = bottom_surface
        if top_surface is not None:
            self.top_surface = top_surface
        if elevation is not None:
            self.elevation = elevation
        else:
            self.elevation = top_surface - bottom_surface
        assert self.elevation is not None, "Elevation is not defined"

    def run(self):
        """This method runs the closed STL generator"""


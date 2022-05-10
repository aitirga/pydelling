from pydelling.paraview_processor.filters import BaseFilter
try:
    from paraview.simple import *
except:
    pass


class TableToPointsFilter(BaseFilter):
    """
    This class implements the Table to Points filter
    """
    filter_type: str = "Table_to_points"
    counter: int = 0

    def __init__(self, filename, name, **params):
        super().__init__(name=name)
        TableToPointsFilter.counter += 1
        self.filter = TableToPoints(Input=str(filename))
        for param in params:
            setattr(self.filter, param, params[param])

    def table2points_columns(self, x_column, y_column, z_column):
        """
        This method sets de xyz columns
        """
        self.filter.Xcolumn = x_column
        self.filter.Ycolumn = y_column
        self.filter.Zcolumn = z_column

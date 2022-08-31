import logging
from typing import Dict

import pandas as pd

from pydelling.paraview_processor.filters import VtkFilter, \
    BaseFilter, CalculatorFilter, IntegrateVariablesFilter, PlotOverLineFilter, XDMFFilter, CellDataToPointDataFilter, \
    ClipFilter, StreamTracerWithCustomSourceFilter, SaveDataFilter, AppendArcLengthFilter, TableToPointsFilter, \
    CsvReaderFilter

logger = logging.getLogger(__name__)
try:
    from paraview.simple import *
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview import servermanager as sm
except:
    logger.warning("Paraview python implementation is not properly set-up")

class ParaviewProcessor:
    """
    This class provides the framework to read data from a VTK file and do different postprocessing steps
    """
    current_array: None
   # calculator: Calculator
    pipeline: Dict[str, BaseFilter] = {}
    vtk_data_counter: int = 0
    xdmf_data_counter: int = 0
    csv_data_counter: int = 0

    def add_vtk_file(self, path, name=None) -> VtkFilter:
        """
        Reads a given vtk file. This is done by adding an instance of the LegacyVTKReader class to the pipeline.
        Args:
            filename: The path of the vtk file.
            name: A custom name to be given in the pipeline.
        Returns:
            The created LegacyVTKReader instance
        """
        pipeline_name = name if name else f"vtk_data_{VtkFilter.counter}"
        self.vtk_data_counter += 1
        vtk_filter = VtkFilter(filename=str(path), name=pipeline_name)
        self.pipeline[pipeline_name] = vtk_filter
        logger.info(f"Added VTK file {path} as {vtk_filter.name} object to Paraview processor")
        return vtk_filter

    def add_csv_file(self, path, name=None, coordinate_labels=("x", "y", "z")) -> CsvReaderFilter:
        """
        Reads a given csv file. A TableToPoints filter is applied to the read csv file
        Args:
            filename: The path of the vtk file.
            name: A custom name to be given in the pipeline.
            coordinate_labels: The names of the x, y and z variables used on the csv file
        Returns:
            The TableToPoints filter containing the csv data
        """
        pipeline_name = name if name else f"vtk_data_{VtkFilter.counter}"
        self.csv_data_counter += 1
        csv_reader = CsvReaderFilter(filename=str(path), name=pipeline_name, coordinate_labels=coordinate_labels)
        self.pipeline[pipeline_name] = csv_reader
        logger.info(f"Added CSV file {path} as {csv_reader.name} object to Paraview processor")
        return csv_reader

    def add_xdmf_file(self, path, name=None) -> XDMFFilter:
        """
        Reads a given xdmf file.
        Args:
            filename: The path of the xdmf file.
            name: A custom name to be given in the pipeline.
        Returns:
            The created XDMFReader instance
        """
        pipeline_name = name if name else f"xdmf_data_{XDMFFilter.counter}"
        self.vtk_data_counter += 1
        pv_filter = XDMFFilter(filename=str(path), name=pipeline_name)
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added XDMF file {path} as {pv_filter.name} object to Paraview processor")
        return pv_filter

    def add_calculator(self, input_filter, function='', name=None, output_array_name='Results', *args, **kwargs) -> CalculatorFilter:
        """
        Adds a calculator filter to a dataset
        Returns:
            The Calculator object
        """
        pipeline_name = name if name else f"calculator_{CalculatorFilter.counter}"
        calculator_filter = CalculatorFilter(input_filter=self.process_input_filter(filter=input_filter),
                                           function=function,
                                           name=pipeline_name,
                                           output_array_name=output_array_name, *args, **kwargs)
        self.pipeline[pipeline_name] = calculator_filter
        logger.info(f"Added calculator filter based on {self.get_input_object_name(input_filter)} as {calculator_filter.name} object to Paraview processor")
        return calculator_filter

    def add_cell_data_to_point_data(self, input_filter, name=None) -> CellDataToPointDataFilter:
        """
        Adds a cell data to point data filter to a dataset
        Returns:
            The CellDataToPointDataFilter object
        """
        pipeline_name = name if name else f"cell_data_to_point_data{CellDataToPointDataFilter.counter}"
        pv_filter = CellDataToPointDataFilter(input_filter=self.process_input_filter(filter=input_filter),
                                           name=pipeline_name,
                                                      )
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added cell_data_to_point_data filter based on {self.get_input_object_name(input_filter)} as {pv_filter.name} object to Paraview processor")
        return pv_filter

    def add_clip(self, input_filter, name=None, *args, **kwargs) -> ClipFilter:
        """
        Adds a clip filter to a dataset
        Returns:
            The ClipFilter object
        """
        pipeline_name = name if name else f"clip_{ClipFilter.counter}"
        pv_filter = ClipFilter(input_filter=self.process_input_filter(filter=input_filter),
                                           name=pipeline_name, *args, **kwargs
                              )
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added clip filter based on {self.get_input_object_name(input_filter)} as {pv_filter.name} object to Paraview processor")
        return pv_filter

    def add_table_to_points(self, path, name=None) -> TableToPointsFilter:
        """
        Adds a table to points filter to a dataset
        Returns:
            The TableToPointsFilter object
        """
        pipeline_name = name if name else f"table_to_points_{TableToPointsFilter.counter}"
        pv_filter = TableToPointsFilter(filename=str(path), name=pipeline_name)
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added table to points filter based on {path} as {pv_filter.name} object to Paraview processor")
        return pv_filter


    def add_stream_tracer_with_custom_source(self, input_filter, seed_source, name=None) -> StreamTracerWithCustomSourceFilter:
        """
        Adds a stream tracer filter to a custom source
        Returns:
            The StreamTracerWithCustomSourceFilter object
        """
        pipeline_name = name if name else f"stream_tracer_with_custom_source_{StreamTracerWithCustomSourceFilter.counter}"
        pv_filter = StreamTracerWithCustomSourceFilter(input_filter=self.process_input_filter(filter=input_filter),
                                           seed_source=self.process_input_filter(filter=seed_source),
                                           name=pipeline_name,
                                                        )
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added stream tracer filter based on {self.get_input_object_name(input_filter)} as {pv_filter.name} object to Paraview processor")
        return pv_filter

    def add_append_arc_length(self, input_filter, name=None) -> AppendArcLengthFilter:
        """
        Adds an append arc-length filter to a dataset
        Returns:
            The AppendArcLengthFilter object
        """
        pipeline_name = name if name else f"append_arc_length_{AppendArcLengthFilter.counter}"
        pv_filter = AppendArcLengthFilter(input_filter=self.process_input_filter(filter=input_filter),
                                           name=pipeline_name
                              )
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added append arc length filter based on {self.get_input_object_name(input_filter)} as {pv_filter.name} object to Paraview processor")
        return pv_filter

    def add_save_data(self, path, proxy, PointDataArrays, CellDataArrays, name=None) -> SaveDataFilter:
        """
        Writes a csv file.
        Args:
            filename: The path of the xdmf file.
            name: A custom name to be given in the pipeline.
        Returns:
            The created SaveDataFilter object
        """
        pipeline_name = name if name else f"save_data_{SaveDataFilter.counter}"
        pv_filter = SaveDataFilter(filename=str(path), proxy=self.process_input_filter(filter=proxy),
                                   point_data_arrays=PointDataArrays, cell_data_arrays=CellDataArrays,
                                   name=pipeline_name)
        self.pipeline[pipeline_name] = pv_filter
        logger.info(f"Added csv file based on {self.get_input_object_name(proxy)} as {pv_filter.name} object to Paraview processor")
        return pv_filter


    def add_integrate_variables(self, input_filter, name=None, divide_cell_data_by_volume=False) -> IntegrateVariablesFilter:
        """
        Adds the integrate_variables filter to a dataset
        Returns:
            An IntegrateVariablesFilter object
        """
        pipeline_name = name if name else f"integrate_variables_{IntegrateVariablesFilter.counter}"
        integrate_variables_filter = IntegrateVariablesFilter(input_filter=self.process_input_filter(filter=input_filter),
                                                              name=pipeline_name,
                                                              divide_cell_data_by_volume=divide_cell_data_by_volume
                                                              )
        self.pipeline[pipeline_name] = integrate_variables_filter
        logger.info(
            f"Added integrate_variables filter based on {self.get_input_object_name(input_filter)} as {integrate_variables_filter.name} object to Paraview processor")
        return integrate_variables_filter

    def add_plot_over_line(self, input_filter, name=None, point_1=None, point_2=None, line_resolution=None) -> PlotOverLineFilter:
        """
        Adds the plot_over_line filter to a dataset
        Returns:
            A [PlotOverLineFilter][pydelling/paraview_processor/filters/PlotOverLineFilter.py] object
        """
        pipeline_name = name if name else f"plot_over_line_{PlotOverLineFilter.counter}"
        plot_over_line_filter = PlotOverLineFilter(input_filter=self.process_input_filter(filter=input_filter),
                                                   name=pipeline_name,
                                                   point_1=point_1,
                                                   point_2=point_2,
                                                   n_line=line_resolution,
                                                   )
        self.pipeline[pipeline_name] = plot_over_line_filter
        logger.info(
            f"Added plot_over_line_filter filter based on {self.get_input_object_name(input_filter)} as {plot_over_line_filter.name} object to Paraview processor")
        return plot_over_line_filter

    # Utility methods
    def plot_over_z_given_xy_point(self, dataset: BaseFilter, x_point, y_point, line_resolution=None) -> PlotOverLineFilter:
        """
        This method interpolates the dataset over a line going in the z-axis given XY coordinates in space
        Args:
            x_point: X coordinate
            y_point: Y coordinate

        Returns:
            A plot_over_line object
        """
        # Compute z_min and z_max values of the dataset
        point_1 = [x_point, y_point, dataset.z_min]
        point_2 = [x_point, y_point, dataset.z_max]
        plot_over_line_filter = PlotOverLineFilter(input_filter=self.process_input_filter(filter=dataset),
                                                   name="plot_over_z_given_xy_point",
                                                   point_1=point_1,
                                                   point_2=point_2,
                                                   n_line=line_resolution,
                                                   )
        return plot_over_line_filter

    def aperture_given_xy_point(self, dataset: BaseFilter, x_point, y_point, variable=None, target_value=1.0, threshold=0.45, line_resolution=100) -> float:
        """
        This method computes the aperture at a given position in the XY plane.
        Args:
            x_point: X coordinate
            y_point: Y coordinate
            variable: The variable to consider that indicates the aperture
            target_value: Value of the variable when the fracture is open
            threshold: Value of the maximum variation from the target_value |line[variable] - target_value| < threshold is assumed
            line_resolution: Resolution of the line interpolation that is being used
        Returns:
            The value of the aperture at a given point
        """
        line_interpolation = self.plot_over_z_given_xy_point(x_point=x_point,
                                                             y_point=y_point,
                                                             line_resolution=line_resolution,
                                                             dataset=dataset,
                                                             )
        line_interpolation_point_data = line_interpolation.point_data
        dataset_mesh_points = dataset.mesh_points
        if variable:
            # If a target variable is defined,
            line_interpolation = line_interpolation_point_data[abs(line_interpolation_point_data[variable] - target_value) < threshold]
            z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation.index]["z"]
            if len(z_points) > 0:
                aperture = abs(z_points.max() - z_points.min())
            else:
                aperture = 0.0
            return aperture

        if not variable:
            # Calculate directly the aperture
            z_points: pd.Series = dataset_mesh_points.iloc[line_interpolation_point_data.index]["z"]
            if len(z_points) > 0:
                aperture = abs(z_points.max() - z_points.min())
            else:
                aperture = 0.0
            return aperture




    def print_pipeline(self) -> str:
        """
        Creates a simple representation of the current working filters and objects
        Returns:
            A string containing the pipeline structure
        """
        identation_level: int = 0
        print_string: str = f""
        print_string += "Paraview filters:\n"
        identation_level += 1
        for pipeline_element in self.pipeline:
            # if type(self.pipeline[pipeline_element]) == dict:
            #     print_string = self.print_pipeline_block(pipeline_dict=self.pipeline[pipeline_element],
            #                               starting_identation_level=identation_level+1,
            #                               output_string=print_string)
            #     continue
            identation = "\t"*identation_level
            print_string += f"{identation}- {pipeline_element} [{self.pipeline[pipeline_element].filter_type}]\n"
        print(print_string)
        return print_string

    def print_pipeline_block(self, pipeline_dict: Dict, output_string:str,  starting_identation_level: int = 0):
        for pipeline_element in pipeline_dict:
            if type(pipeline_dict[pipeline_element]) == dict:
                output_string = self.print_pipeline_block(pipeline_dict=pipeline_dict[pipeline_element],
                                          starting_identation_level=starting_identation_level+1,
                                          output_string=output_string)
                continue
            identation = "\t"*starting_identation_level
            output_string += f"{identation}- {pipeline_element}\n"
        return output_string

    def get_object(self, name):
        """
        Get a given object from the pipeline
        Args:
            name: The name of the object to obtain

        Returns:
            The requested object
        """
        return self.pipeline[name]

    def process_input_filter(self, filter) -> object:
        """
        Processes the filter object and returns the proper datatype
        Args:
            input_filter: an object refering to an existing filter

        Returns:
            A proper Paraview object
        """
        if type(filter) == str:
            # Assume the filter specifies the name of the pipeline
            return self.pipeline[filter].filter
        else:
            return filter.filter

    @staticmethod
    def get_input_object_name(filter) -> str:
        """
        Gets the name of a given filter
        Args:
            filter: filter to be processed

        Returns:
            String containing the filter's name
        """
        if type(filter) == str:
            return filter
        else:
            return filter.name




    def __repr__(self):
        return self.print_pipeline()

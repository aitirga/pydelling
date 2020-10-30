import numpy as np
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from paraview import servermanager as sm
from typing import Dict
import logging

from PyFLOTRAN.paraview_processor.filters import VtkFilter, BaseFilter

logger = logging.getLogger(__name__)


class ParaviewProcessor:
    """
    This class provides the framework to read data from a VTK file and do different postprocessing steps
    """
    current_array: None
    calculator: Calculator
    pipeline: Dict[str, BaseFilter] = {}
    vtk_data_counter: int = 0

    def add_vtk_file(self, path, name=None):
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
        self.pipeline[pipeline_name] = VtkFilter(filename=str(path), name=pipeline_name)
        return self.pipeline[pipeline_name]

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

    def add_calculator(self, input=None, function=''):
        """
        Adds a calculator filter to a dataset
        Returns:
            The Calculator object
        """
        input = input if input else self.current_array
        self.calculator = Calculator(Input=input)
        self.calculator.Function = function

        self.current_array = self.calculator
        return self.current_array

    def __repr__(self):
        return self.print_pipeline()

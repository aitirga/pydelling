import logging

import numpy as np

from pydelling.config import config
from .BaseWriter import BaseWriter

logger = logging.getLogger(__name__)


class OpenFoamVariableWriter(BaseWriter):
    """This class creates an OpenFOAM variable data file"""
    def __init__(self, filename: str = None, header=None, outer=None, data=None, configuration_dict=None, *args, **kwargs):
        """
        A correct set of header/outer needs to be provided to the class.

        It can be automatically used from the config file by creating an instance called "open_foam_variable_writer" for the settings.

        Args:
            filename: Name of the output file
            header: Dictionary containing the header variables
            outer: Dictioanary containing the end variables
            data: Data to be set into the OpenFOAM file
        """
        if configuration_dict:
            self.header = configuration_dict.header
            self.outer = configuration_dict.outer
            self.filename = configuration_dict.filename
        else:
            self.header = header if header else config.open_foam_variable_writer.header
            self.outer = outer if outer else config.open_foam_variable_writer.outer
            self.filename = filename if filename else config.open_foam_variable_writer.filename
        self.data = data
        super().__init__(filename=self.filename, *args, **kwargs)

    def run(self, *args, **kwargs):
        """
        Writes the data into an OpenFOAM variable format
        """
        logger.info(f"Writing data to {self.filename}")
        with open(self.filename, "w") as self.output_file:
            self.write_header()
            self.write_data()
            self.write_outer()

    def write_header(self):
        """
        Writes the header of the file
        """
        self.output_file.write(f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\    /   O peration     | Website:  https://openfoam.org
    \\\  /    A nd           | Version:  7
     \\\/     M anipulation  |
\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {self.header.field_type};
    location    "{self.header.location}";
    object      {self.header.object};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      {self.header.dimensions};
internalField   {self.header.data_type} {self.header.data_structure}\n""")

    def write_data(self):
        self.output_file.write(f"{len(self.data)}\n")
        self.output_file.write("(\n")
        for data_element in self.data:
            if type(data_element) == np.float64 or type(data_element) == np.float32:
                self.output_file.write(f"{data_element}\n")
            elif len(data_element) == 4:
                self.output_file.write(f"{data_element[3]}\n")
            else:
                logger.error("There is an error provided the data element vector")
                raise ValueError("There was an error providing the data element")

        self.output_file.write(")\n")
        self.output_file.write(";\n")

    def write_outer(self):
        self.output_file.write(f"""boundaryField\n{{\n""")
        for region in self.outer.boundary_fields:
            region_dict = self.outer.boundary_fields[region]
            self.output_file.write(f"""\t{region}\n\t{{\n\t\ttype\t{region_dict["type"]};\n\t}}\n""")
        self.output_file.write("}\n")
        self.output_file.write("\n")
        self.output_file.write("// ************************************************************************* //")

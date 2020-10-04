import os
import h5py
from .BaseWriter import BaseWriter
from pathlib import Path
from PyFLOTRAN.config import config
import logging

logger = logging.getLogger(__name__)


class OpenFoamVariableWriter(BaseWriter):
    def __init__(self, filename=None, header=None, outer=None, data=None, *args, **kwargs):
        self.header = header if header else config.open_foam_variable_writer.header
        self.outer = outer if outer else config.open_foam_variable_writer.outer
        self.filename = filename if filename else config.open_foam_variable_writer.filename
        self.data = data
        super().__init__(filename=self.filename, *args, **kwargs)

    def run(self, *args, **kwargs):
        """Writes the data into an OpenFOAM variable format

        Returns
        -------

        """
        logger.info(f"Writing data to {self.filename}")
        with open(self.filename, "w") as self.output_file:
            self.write_header()
            self.write_data()
            self.write_outer()

    def write_header(self):
        """Writes the header of the file

        Returns
        -------

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
    class       {config.open_foam_variable_writer.header.field_type};
    location    "{config.open_foam_variable_writer.header.location}";
    object      {config.open_foam_variable_writer.header.object};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      {config.open_foam_variable_writer.header.dimensions};
internalField   {config.open_foam_variable_writer.header.data_type} {config.open_foam_variable_writer.header.data_structure}\n""")

    def write_data(self):
        self.output_file.write(f"{len(self.data)}\n")
        self.output_file.write("(\n")
        for data_element in self.data:
            self.output_file.write(f"{data_element[3]}\n")
        self.output_file.write(")\n")
        self.output_file.write(";\n")

    def write_outer(self):
        self.output_file.write(f"""boundaryField\n{{\n""")
        for region in config.open_foam_variable_writer.outer.boundary_fields:
            region_dict = config.open_foam_variable_writer.outer.boundary_fields[region]
            self.output_file.write(f"""\t{region}\n\t{{\n\t\ttype\t{region_dict["type"]};\n\t}}\n""")
        self.output_file.write("}\n")
        self.output_file.write("\n")
        self.output_file.write("// ************************************************************************* //")

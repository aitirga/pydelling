"""
Author: Aitor Iraola
Date: 030920
Description: This script postprocesses a given PFLOTRAN simulation carried out using an unstructured explicit mesh and the methodology developed by Amphos 21
"""
from pathlib import Path
import h5py
import os
import shutil
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class PflotranPostprocessing:
    """This class manages the structure of the output files"""
    def __init__(self, dt=0.2, unit="y"):
        self.dt = dt
        self.unit = unit
        self.output_times = []
        self.output_units = []
        self.project_name = None

    def run(self):
        "Method that runs the post-processing engine"
        self.find_directories()
        self.find_output_files()
        self.find_domain_file()
        self.find_input_file()
        if len(self.output_vtk_vel_files) != 0:
            self.add_velocity_to_hdf5()
        self.find_attributes()
        self.export_xmf()
        self.copy_domain_file()

    def find_output_files(self):
        """This method searches the current directory and the directory called "./output-hdf5" to find PFLOTRAN output files"""
        # First, see if the directory contains the output files
        if self.current_directory.stem == "output-hdf5":
            self.output_h5_files = list(self.current_directory.glob("*.h5"))
            self.output_directory = self.current_directory
            assert len(self.output_h5_files) > 0, "Output hdf5 files couldn't be found"
        else:
            self.output_h5_files = list(self.root_directory.glob("*.h5"))
            self.output_directory = self.current_directory
            if len(self.output_h5_files) == 0:
                # Check the output-hdf5 folder
                output_folder = self.current_directory / "output-hdf5"
                self.output_h5_files = list(output_folder.glob("*.h5"))
                # assert len(self.output_h5_files) > 0, "Output hdf5 files couldn't be found"
                self.output_directory = self.current_directory / "output-hdf5"
        # Take out the domain hdf5 file from the output file list
        # Set the project name
        domain_file = [file for file in self.output_h5_files if "domain" in file.name]
        assert len(domain_file) > 0, "Domain file couldn't be found"
        self.output_h5_files.remove(domain_file[0])
        self.project_name = self.output_h5_files[0].stem.split('-')[0]
        # Rename domain file using the project name
        domain_file[0].rename(self.output_directory / f"{self.project_name}-domain.h5")
        self.output_h5_files = sorted(self.output_h5_files)
        # Process the VTK files
        if self.current_directory.stem == "output-vtk":
            self.output_vtk_vel_files = list(self.current_directory.glob("*vel*.vtk"))
            self.output_vtk_directory = self.current_directory
            assert len(self.output_vtk_vel_files) > 0, "Output hdf5 files couldn't be found"
        else:
            self.output_vtk_vel_files = list(self.root_directory.glob("*vel*.vtk"))
            self.output_vtk_directory = self.current_directory
            if len(self.output_vtk_vel_files) == 0:
                # Check the output-hdf5 folder
                output_folder = self.current_directory / "output-vtk"
                self.output_vtk_vel_files = list(output_folder.glob("*vel*.vtk"))
                # assert len(self.output_vtk_files) > 0, "Output vtk files couldn't be found"
                self.output_vtk_directory = self.current_directory / "output-vtk"
        self.output_vtk_vel_files = sorted(self.output_vtk_vel_files)

    def find_directories(self):
        """Test """
        self.current_directory = Path.cwd()
        if self.current_directory.stem == "output-hdf5":
            self.root_directory = self.current_directory.parent
        elif self.current_directory.glob("*.in"):
            self.root_directory = self.current_directory
        elif self.current_directory.stem == "input_files":
            self.root_directory = self.current_directory.parent
        else:
            raise FileNotFoundError("Root directory couldn't be found")
        self.input_directory = self.root_directory / "input_files"

    def find_domain_file(self):
        """This method finds the domain file"""
        if len(list(self.input_directory.glob("*-domain.h5"))) > 0:
            self.domain_file = list(self.input_directory.glob("*-domain.h5"))[0]
        else:
            self.domain_file = list(self.current_directory.glob("*-domain.h5"))[0]
        return self.domain_file

    def find_input_file(self):
        self.input_file = None
        try:
            self.input_file = list(self.input_directory.glob("*.in"))[0]
        except:
            logger.warning("Input file couldn't be found")
        return self.input_file

    def add_velocity_to_hdf5(self):
        for i in range(0, self.n_output_files):
            with h5py.File(self.output_h5_files[i], "r+") as h5_file:
                vtk_filename = self.output_vtk_vel_files[i]
                vtk_file_dict = self.read_vtk_file(filename=vtk_filename)
                for var in vtk_file_dict:
                    if var == "Material_ID":
                        continue
                    hdf5_group = list(h5_file.values())[0]
                    var_names = [name for name in hdf5_group]
                    if not var in var_names:
                        hdf5_group.create_dataset(name=var, data=vtk_file_dict[var]["data"])

    def find_attributes(self):
        # Find number of cells and number of vertices
        with h5py.File(self.domain_file, "r") as file:
            self.n_vertices = len(file["Domain"]["Vertices"])
            self.pflotran_vertices = len(file["Domain"]["Cells"])
        # Find output variables
        with h5py.File(self.output_h5_files[0], "r") as hdf5_file:
            hdf5_group = list(hdf5_file.values())[0]
            self.output_variables = [item[0] for item in hdf5_group.items()]
            self.n_cells = len(hdf5_group[self.output_variables[0]])
        for hdf5_file in self.output_h5_files:
            if 'domain' in hdf5_file.name:
                # Delete the domain file from the list
                self.output_h5_files.remove(hdf5_file)
                continue

            with h5py.File(hdf5_file, "r") as hdf5_file:
                attribute_list = list(hdf5_file.items())[0][0].split()
                self.output_times.append(float(attribute_list[2]))
                self.output_units.append(attribute_list[3])

    def copy_domain_file(self):
        # check if file exists and delete otherwise
        # if list(self.output_directory.glob("*-domain.h5")):
        #     os.remove(list(self.output_directory.glob("*-domain.h5"))[0])
        # target_file = self.output_directory / "{}-domain.h5".format(self.project_name)
        # domain_file = str(self.domain_file)
        # shutil.copy(domain_file, str(target_file))
        pass

    def export_xmf(self):
        # Input variables
        dt = self.dt  # time interval between individual result files
        file_name = self.project_name
        Nelements = self.n_cells
        Nvertices = self.pflotran_vertices - self.n_cells
        Nvertices_pflotran = self.n_vertices


        # Script to generate the files
        for i in range(0, self.n_output_files):
            self.name = "{}-{:03d}.xmf".format(file_name, i)
            self.time = dt * i
            name_h5 = "{}-{:03d}.h5".format(file_name, i)
            self.export_file = self.output_directory
            xdmf_file = self.output_directory / self.name
            with open(str(xdmf_file), 'w') as self.output_file:
                self.output_file.write('<?xml version="1.0" ?>\n')
                self.output_file.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
                self.output_file.write('<Xdmf>\n')
                self.output_file.write('\t<Domain>\n')
                self.output_file.write('\t<Grid Name="Mesh">\n')
                self.output_file.write('\t\t<Time Value = "{:1.5E}" />\n'.format(self.output_times[i]))
                self.output_file.write('\t\t<Topology Type="Mixed" NumberOfElements="%s" >\n' % Nelements)
                self.output_file.write('\t\t\t<DataItem Format="HDF" DataType="Int" Dimensions="%s">\n' % (Nelements + Nvertices))
                self.output_file.write('\t\t\t\t %s-domain.h5:/Domain/Cells\n' % file_name)
                self.output_file.write('\t\t\t</DataItem>\n')
                self.output_file.write('\t\t</Topology>\n')
                self.output_file.write('\t\t<Geometry GeometryType="XYZ">\n')
                self.output_file.write('\t\t\t<DataItem Format="HDF" Dimensions="%s 3">\n' % Nvertices_pflotran)
                self.output_file.write('\t\t\t\t %s-domain.h5:/Domain/Vertices\n' % file_name)
                self.output_file.write('\t\t\t</DataItem>\n')
                self.output_file.write('\t\t</Geometry>\n')
                for output_variable in self.output_variables:
                    self.export_attribute(output_variable, i)
                self.output_file.write('\t</Grid>\n')
                self.output_file.write('\t</Domain>\n')
                self.output_file.write('</Xdmf>\n')

    def export_attribute(self, var, i):
        self.output_file.write('\t\t<Attribute Name="%s" AttributeType="Scalar"  Center="Cell">\n' % var)
        self.output_file.write('\t\t\t<DataItem Dimensions="%s 1" Format="HDF">\n' % self.n_cells)
        self.output_file.write('\t\t\t\t{}-{:03d}.h5:/{:4d} Time  {:1.5E} {}/{}\n'.format(self.project_name, i, i, self.output_times[i], self.output_units[i], var))
        self.output_file.write('\t\t\t</DataItem>\n')
        self.output_file.write('\t\t</Attribute>\n')

    @staticmethod
    def read_vtk_file(filename) -> Dict:
        """This method reads an VTK file and returns the data"""
        # Process vtk file
        print("Processing {} VTK file".format(filename))
        data_dict = {}
        temp_array = []
        start_reading = False
        # filename = "G:/My Drive/dev/2020-project_repos/rwm_repo/scripts/3-post_processing/B-VTK_velocity/output-vtk/test-trial.vtk"
        with open(filename, "r") as vtk_file:
            for line in vtk_file.readlines():
                split_line = line.split()
                if not split_line:
                    start_reading = False
                if "CELL_DATA" in split_line:
                    n_cells = split_line[1]
                if "SCALARS" in split_line:
                    data_dict[split_line[1]] = {"name": split_line[1],
                                                "n_cells": n_cells,
                                                "data": []
                                                }
                    try:
                        data_dict[current_reading]["data"] = np.concatenate(temp_array)
                        data_dict[current_reading]["shape"] = data_dict[current_reading]["data"].shape
                    except:
                        pass
                    start_reading = False
                    current_reading = split_line[1]
                    print("Starting to read variable {}".format(current_reading))
                if "LOOKUP_TABLE" in split_line:
                    start_reading = True
                    temp_array = []
                    continue
                if start_reading:
                    temp_array.append(np.array(split_line).astype(np.float))
            data_dict[current_reading]["data"] = np.concatenate(temp_array)
            data_dict[current_reading]["shape"] = np.concatenate(temp_array).shape
        return data_dict

    @property
    def n_output_files(self):
        return len(self.output_h5_files)

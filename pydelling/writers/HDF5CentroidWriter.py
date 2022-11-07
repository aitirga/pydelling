import logging
import os
from pathlib import Path

import h5py
import numpy as np

from .BaseWriter import BaseWriter

logger = logging.getLogger(__name__)

class HDF5CentroidWriter(BaseWriter):
    def run(self, filename=None, remove_if_exists=True, include_cell_id=True):
        if filename is not None:
            self.filename = filename
        if remove_if_exists:
            try:
                os.remove(self.filename)
            except FileNotFoundError as ef:
                print("Nothing to overwrite!")

        if self.check_data():
            if not os.path.exists(self.filename):
                h5temp = h5py.File(self.filename, "w")
                h5temp.close()
            with h5py.File(self.filename, "r+") as h5temp:
                h5temp.create_dataset(self.var_name, data=self.data)
                if include_cell_id:
                    cell_id = np.array([index + 1 for index in range(len(self.data))])
                    h5temp.create_dataset("Cell Ids", data=cell_id)
        else:
            print("Couldn't find data to dump!")

    def write_anisotropic_dataset(self,
                                  dataset_x,
                                  dataset_y,
                                  dataset_z,
                                  filename=None,
                                  remove_if_exists=True,
                                  var_name=None,
                                  ):
        """
        Writes anisotropic dataset to HDF5
        Args:
            filename: name of the file
            dataset_x: dataset containing data in the x-direction
            dataset_y: dataset containing data in the y-direction
            dataset_z: dataset containing data in the z-direction
            remove_if_exists: removes file if exists
        """
        filename_path = Path(filename if filename else self.filename)
        logger.info(f"Writing anisotropic permeability to {filename_path}")
        if remove_if_exists:
            if filename_path.exists():
                filename_path.unlink()
        with h5py.File(filename_path, "w") as h5_temp:
            var_name_write = var_name if var_name else self.var_name
            h5_temp.create_dataset(f"{var_name_write}X", data=dataset_x)
            h5_temp.create_dataset(f"{var_name_write}Y", data=dataset_y)
            h5_temp.create_dataset(f"{var_name_write}Z", data=dataset_z)
            cell_id = np.array([index + 1 for index in range(len(dataset_x))])
            h5_temp.create_dataset("Cell Ids", data=cell_id)

    def write_full_tensor_dataset(self,
                                  dataset_x,
                                  dataset_xy,
                                  dataset_xz,
                                  dataset_y,
                                  dataset_yz,
                                  dataset_z,
                                  filename=None,
                                  remove_if_exists=True,
                                  var_name=None,
                                  ):
        """
        Writes anisotropic dataset to HDF5
        Args:
            filename: name of the file
            dataset_x: dataset containing data in the x-direction
            dataset_xy: dataset containing data in the xy-direction
            dataset_xz: dataset containing data in the xz-direction
            dataset_y: dataset containing data in the y-direction
            dataset_yz: dataset containing data in the yz-direction
            dataset_z: dataset containing data in the z-direction
            remove_if_exists: removes file if exists
        """
        filename_path = Path(filename if filename else self.filename)
        logger.info(f"Writing full tensor permeability to {filename_path}")
        if remove_if_exists:
            if filename_path.exists():
                filename_path.unlink()
        with h5py.File(filename_path, "w") as h5_temp:
            var_name_write = var_name if var_name else self.var_name
            h5_temp.create_dataset(f"{var_name_write}X", data=dataset_x)
            h5_temp.create_dataset(f"{var_name_write}XY", data=dataset_xy)
            h5_temp.create_dataset(f"{var_name_write}XZ", data=dataset_xz)
            h5_temp.create_dataset(f"{var_name_write}Y", data=dataset_y)
            h5_temp.create_dataset(f"{var_name_write}YZ", data=dataset_yz)
            h5_temp.create_dataset(f"{var_name_write}Z", data=dataset_z)
            cell_id = np.array([index + 1 for index in range(len(dataset_x))])
            h5_temp.create_dataset("Cell Ids", data=cell_id)


    def write_dataset(self,
                      dataset,
                      filename=None,
                      remove_if_exists=True,
                      var_name=None,
                      ):
        """
        Writes isotropic dataset to HDF5
        Args:
            filename: name of the file
            dataset_x: output dataset
            remove_if_exists: removes file if exists
        """
        filename_path = Path(filename if filename else self.filename)
        logger.info(f"Writing anisotropic permeability to {filename_path}")
        if remove_if_exists:
            if filename_path.exists():
                filename_path.unlink()
        with h5py.File(filename_path, "w") as h5_temp:
            var_name_write = var_name if var_name else self.var_name
            h5_temp.create_dataset(f"{var_name_write}", data=dataset)
            cell_id = np.array([index + 1 for index in range(len(dataset))])
            h5_temp.create_dataset("Cell Ids", data=cell_id)
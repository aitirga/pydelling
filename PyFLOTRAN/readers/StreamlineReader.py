"""
Centroid file reader
"""
import numpy as np
from .BaseReader import BaseReader
from PyFLOTRAN.config import config
from PyFLOTRAN.utils.utils import aperture_from_a_xy_point
from PyFLOTRAN.paraview_processor import ParaviewProcessor
import logging
import pandas as pd
import pickle
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StreamlineReader(BaseReader):
    data: pd.DataFrame
    raw_data: pd.DataFrame
    stream_data: DataFrameGroupBy
    is_aperture_zero: bool = False

    def __init__(self, filename, header=False):
        self.header = None
        super().__init__(filename,
                         header=header)

    def read_file(self, opened_file):
        """
        This method reads an opened file an sets the `self` variables properly to be used throuhgout the class
        Args:
            opened_file: the object containing the opened file
        """
        logger.info(f"Reading streamlines from {self.filename}")
        self.data = pd.read_csv(opened_file)
        self.raw_data = self.data.copy()  # Copy of original data
        self.data = self.data.rename(columns={"Points:0": "x",
                                              "Points:1": "y",
                                              "Points:2": "z",
                                              }
                                     )
        self.stream_data: DataFrameGroupBy = self.data.groupby("SeedIds")

    def compute_arrival_times(self, reason_of_termination=None) -> pd.Series:
        """
        This method computes the arrival times of the streamlines
        Returns:
             A pd.Series object containing the arrival times of the streamlines
        """
        logger.info("Computing arrival times of the streamlines")
        reason_of_termination = reason_of_termination if reason_of_termination else config.streamline_reader.reason_of_termination
        temp_df = self.stream_data
        if reason_of_termination:
            temp_df = temp_df.filter(lambda x: x["ReasonForTermination"].max() == reason_of_termination)
        temp_series: pd.Series = temp_df.groupby("SeedIds").max()["IntegrationTime"]
        return temp_series

    def compute_length_streamlines(self, reason_of_termination=None) -> pd.Series:
        """
        This method computes the length of the streamlines
        Returns:
             A pd.Series object containing the length of the streamlines
        """
        logger.info("Computing length of the streamlines")
        reason_of_termination = reason_of_termination if reason_of_termination else config.streamline_reader.reason_of_termination
        temp_df = self.stream_data
        if reason_of_termination:
            temp_df = temp_df.filter(lambda x: x["ReasonForTermination"].max() == reason_of_termination)
        temp_series: pd.Series = temp_df.groupby("SeedIds").max()["arc_length"]
        return temp_series

    def compute_beta(self, aperture_field: str = None):
        """
        This method computes beta values for each streamline
        Returns:
            A pd.Series object containing the streamline info with the beta column added
        """
        aperture_field_file = aperture_field if aperture_field else config.beta_integrator.aperture_field_file if config.beta_integrator.aperture_field_file else None
        assert aperture_field_file, "Define a file containing an aperture field matrix"
        # Read aperture field
        with open(aperture_field_file, "rb") as opened_file:
            aperture_field = pickle.load(opened_file)
        logger.info(f"Aperture field has been loaded from {aperture_field_file}")
        logger.info("Computing beta values for the streamlines")
        stream_beta = []
        for stream in tqdm(self.stream_data.groups):
            stream_data = self.stream_data.get_group(stream)
            stream_beta_value = self.integrate_beta(stream=stream_data, aperture_field=aperture_field)
            if stream_beta_value == 0.0:
                continue
            if self.is_aperture_zero:
                self.is_aperture_zero = False
                continue
            else:
                stream_beta.append(stream_beta_value)
            # print(f"Computed beta for stream {stream}")
        print(stream_beta)
        return pd.DataFrame(np.array(stream_beta))

    def integrate_beta(self, stream: pd.DataFrame, aperture_field: np.ndarray):
        """
        This method integrates a given streamline to compute the value of beta
        Args:
            stream: A pandas DataFrame containing the data of a given streamline
            aperture_field: A matrix containing the information of the aperture field
        Returns:

        """
        beta = 0.0
        aperture_field_nx = aperture_field.shape[1]
        aperture_field_ny = aperture_field.shape[0]
        previous_integration_time = 0.0
        for index, fragment in stream.iterrows():
            # aperture = aperture_from_a_xy_point(x_point=)
            # Nearest neighbour
            index_x = int(np.floor(fragment["x"] / config.beta_integrator.dimension_x * aperture_field_nx))
            index_y = int(np.floor(fragment["y"] / config.beta_integrator.dimension_y * aperture_field_ny))
            index_row = aperture_field_ny - index_y - 1
            index_column = index_x
            if index_row == aperture_field_ny:
                index_row -= 1
            if index_column == aperture_field_nx:
                index_column -= 1
            # print(fragment["x"], fragment["y"], index_x, index_y, index_row, index_column)
            # print(index_row, index_column, index_x, index_y)
            aperture = aperture_field[index_row, index_column]
            if aperture == 0.0:
                # self.is_aperture_zero = True
                # logger.warning(f"Zero value aperture has been detected on point [{fragment['x']}, {fragment['y']}]")
                continue
            tau = fragment["IntegrationTime"] - previous_integration_time
            beta += 2 * tau / aperture / (365 * 24 * 3600)
            previous_integration_time = fragment["IntegrationTime"]
        return beta

    def get_data(self) -> np.ndarray:
        """
        Outputs the data
        :return: np.ndarray object containing centroid information and variable output
        """
        return self.data.values

    def dump_to_csv(self, output_file, delimiter=","):
        """
        Writes the data into a csv file
        :param output_file:
        :return:
        """
        print(f"Starting dump into {output_file}")
        # self.data.to_csv(output_file, delimiter=delimiter)
        self.data.to_csv(output_file)
        print(f"The data has been properly exported to the {output_file} file")


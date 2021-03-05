"""
Centroid file reader
"""
import numpy as np
from .BaseReader import BaseReader
from PyFLOTRAN.config import config
import logging
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
logger = logging.getLogger(__name__)

class StreamlineReader(BaseReader):
    data: pd.DataFrame
    raw_data: pd.DataFrame
    stream_data: DataFrameGroupBy

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

    # def compute_beta(self):
    #     """
    #     This method computes beta values for each streamline
    #     Returns:
    #         A pd.Series object containing the streamline info with the beta column added
    #     """
    #     logger.info("Computing beta values for the streamlines")
    #     for stream in self.stream_data:
    #         for


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


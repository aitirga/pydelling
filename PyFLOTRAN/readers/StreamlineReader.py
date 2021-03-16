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
        self.data.dropna()
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
            temp_df = temp_df.filter(lambda x: x.max()["ReasonForTermination"] == float(reason_of_termination))
        temp_series: pd.Series = temp_df.groupby("SeedIds").max()["IntegrationTime"]
        return temp_series

    def compute_arrival_times_per_material(self, reason_of_termination=None) -> pd.Series:
        """
        This method computes the arrival times of the streamlines for a particular material
        Returns:
             A pd.Series object containing the arrival times of the streamlines
        """
        logger.info("Computing arrival times of the streamlines per material")
        reason_of_termination = reason_of_termination if reason_of_termination else config.streamline_reader.reason_of_termination
        temp_df = self.stream_data
        if reason_of_termination:
            temp_df = temp_df.filter(lambda x: x["ReasonForTermination"].max() == reason_of_termination)
        temp_df['Material ID'] = temp_df['Material ID'].apply(np.ceil)
        temp_series: pd.Series = temp_df.groupby(["Material ID", "SeedIds"]).max()
        temp_series = temp_series.reset_index()

        # MANUAL
        # group_2 = temp_series.groupby('Material ID').get_group(float(2.0))['IntegrationTime']
        # group_3 = temp_series.groupby('Material ID').get_group(float(3.0))['IntegrationTime']
        # group_4 = temp_series.groupby('Material ID').get_group(float(4.0))['IntegrationTime']
        # group_5 = temp_series.groupby('Material ID').get_group(float(5.0))['IntegrationTime']
        # group_54 = pd.concat([group_5, group_4])
        # group_543 = pd.concat([group_5, group_4, group_3])
        # group_5432 = pd.concat([group_5, group_4, group_3, group_2])
        # return temp_series, group_5, group_54, group_543, group_5432

        # AUTOMATIC
        dic = {}
        dic_group = {}
        for group in temp_series.groupby('Material ID').groups:
            # print(temp_series.groupby('Material ID').get_group(group))
            ngroup = temp_series.groupby('Material ID').get_group(group)['IntegrationTime']
            dic[group] = ngroup
            dic_group[group] = ''

        b = True
        for key in sorted(dic.keys(), reverse=True):
            if b:
                add = dic[key]
                dic_group[key] = add
                b = False
            else:
                add = pd.concat([add, dic[key]])
                dic_group[key] = add

        return temp_series, dic_group

    def compute_initial_velocities(self,
                                   reason_of_termination = None,
                                   normalize=True,
                                   index_df: pd.Series=None,
                                   ):
        """
        This method computes the initial velocities each streamlines 'sees' at the beginning, it can be used to normalize them later on

        Args:
            normalized: Parameter controlling weather the output vector should be normalized by dividing by the maximum velocity valuefdh

        Returns: a vector containing the initial velocities of the streamlines
        """
        logger.info("Computing initial velocities of the streamlines")
        temp_df = self.data
        if index_df is not None:
            seed_ids = index_df.reset_index()["SeedIds"]
            temp_df = temp_df[temp_df["SeedIds"].isin(seed_ids)]
        temp_df = temp_df.groupby("SeedIds").first()
        temp_series = temp_df["U:0"]
        if normalize:
            temp_series /= temp_series.max()
        temp_series[temp_series < 0.0] = 0.0
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
            temp_df = temp_df.filter(lambda x: x["ReasonForTermination"].max() != reason_of_termination)
        temp_series: pd.Series = temp_df.groupby("SeedIds").max()["arc_length"]
        return temp_series

    def compute_beta(self, aperture_field: str = None):
        """
        This method computes beta values for each streamline
        Returns:
            A pd.Series object containing the streamline info with the beta column added
        """
        self.number_of_zero_apertures = 0
        aperture_field_file = aperture_field if aperture_field else config.beta_integrator.aperture_field_file if config.beta_integrator.aperture_field_file else None
        assert aperture_field_file, "Define a file containing an aperture field matrix"
        # Read aperture field
        with open(aperture_field_file, "rb") as opened_file:
            aperture_field = pickle.load(opened_file)
        aperture_field = self.fix_aperture_field(aperture_field)
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
        print(self.number_of_zero_apertures)
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
        previous_aperture = 0.0
        for index, fragment in stream.iterrows():
            # aperture = aperture_from_a_xy_point(x_point=)
            # Nearest neighbour
            index_x = int(np.floor(fragment["x"] / config.beta_integrator.dimension_x * aperture_field_nx))
            index_y = int(np.floor(fragment["y"] / config.beta_integrator.dimension_y * aperture_field_ny))
            index_row = index_y
            index_column = index_x
            if index_row == aperture_field_ny:
                index_row -= 1
            if index_column == aperture_field_nx:
                index_column -= 1

            # Compute tau
            tau = fragment["IntegrationTime"] - previous_integration_time
            # Calculate aperture
            aperture = aperture_field[index_row, index_column]
            if fragment["x"] < 0.001:
                print(f"Aperture: {aperture}, x: {fragment['x']} y: {fragment['y']} index_x: {index_x}, index_y: {index_y}")
            if aperture == 0.0:
                aperture = previous_aperture
                # continue
            previous_aperture = aperture
            try:
                beta += 2 * tau / aperture / (365 * 24 * 3600)
            except:
                self.number_of_zero_apertures += 1
                continue
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

    @staticmethod
    def fix_aperture_field(aperture_matrix):
        aperture_matrix[0, :] = aperture_matrix[1, :]
        aperture_matrix[:, 0] = aperture_matrix[:, 1]
        aperture_matrix[aperture_matrix.shape[0] - 1, :] = aperture_matrix[aperture_matrix.shape[0] - 2, :]
        aperture_matrix[:, aperture_matrix.shape[1] - 1] = aperture_matrix[:, aperture_matrix.shape[1] - 2]
        return aperture_matrix
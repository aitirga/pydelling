from .BaseStudy import BaseStudy
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import logging
from alive_progress import alive_bar
from tqdm import tqdm
from pydelling.utils import create_results_folder
from docker import DockerClient


logger = logging.getLogger(__name__)


class BaseManager(ABC):
    """This class is the base class for all simulation managers. A simulation manager should be able to:
    - Manage study [BaseStudy] objects
    - Run simulations on a given software
    - Read the output status of the simulation
    """
    def __init__(self):
        self.results_folder = None
        self.studies: Dict[str, BaseStudy] = {}

    def add_study(self, study: BaseStudy):
        """This method adds a study to the manager.
        """
        assert isinstance(study, BaseStudy), f"Study must be a object from a class inherited from BaseStudy, not {type(study)}"
        self.studies[study.name] = study

    def run(self,
            studies_folder: str = './studies',
            n_cores: int = 1,
            ):
        """This method runs all the studies.
        """
        self.generate_run_files(studies_folder=studies_folder)
        for study in tqdm(self.studies.values(), desc="Running studies", colour="white"):
            self.run_study(study.name)

    def generate_run_files(self, studies_folder: str = './studies'):
        """This method generates the run files for all the studies.
        """
        self.results_folder = create_results_folder(studies_folder)
        for study in self.studies.values():
            study: BaseStudy
            study.to_file(self.results_folder / study.name)

    @abstractmethod
    def _run_study(self, study_name: str, n_cores: int = 1):
        """This method runs a study.
        """
        logger.info(f"Running study {study_name}")
        pass

    @abstractmethod
    def _run_study_docker(self,
                          study_name: str,
                          docker_image: str,
                          n_cores: int = 1,
                          ):
        """This method runs a study using docker.
        """
        logger.info(f"Running study {study_name} using docker image {docker_image}")


    @abstractmethod
    def _get_study_status(self, study_name: str):
        """This method returns the status of a study.
        """
        pass

    def run_study(self,
                  study_name: str,
                  n_cores: int = 1,
                  docker_image: str = None,
                  ):
        """This method runs a study.
        """
        logger.info(f"Running study {study_name}")
        # Create the study files

        # Run the study
        if docker_image is None:
            self._run_study(study_name, n_cores=n_cores)
        else:
            self._run_study_docker(study_name, docker_image, n_cores=n_cores)


    def get_study_status(self, study_name: str):
        """This method returns the status of a study.
        """
        return self._get_study_status(study_name)

    @property
    def n_studies(self):
        """This method returns the number of studies.
        """
        return len(self.studies)







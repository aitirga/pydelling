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
    def __init__(self, name: str = None):
        self.results_folder = None
        self.studies: Dict[str, BaseStudy] = {}
        self.manager_name = name if name is not None else self.__class__.__name__

    def add_study(self, study: BaseStudy):
        """This method adds a study to the manager.
        """
        assert isinstance(study, BaseStudy), f"Study must be a object from a class inherited from BaseStudy, not {type(study)}"
        self.studies[study.name] = study

    def run(self,
            studies_folder: str = './studies',
            n_cores: int = 1,
            docker_image: str = None,
            ):
        """This method runs all the studies.
        """
        self.generate_run_files(studies_folder=studies_folder)
        for study in tqdm(self.studies.values(), desc="Running studies", colour="white"):
            self.run_study(study, docker_image=docker_image, n_cores=n_cores)

    def generate_run_files(self, studies_folder: str = './studies'):
        """This method generates the run files for all the studies.
        """
        self.results_folder = create_results_folder(studies_folder)
        for study in self.studies.values():
            study: BaseStudy
            study.to_file(self.results_folder / study.name)

    @abstractmethod
    def _run_study(self, study: BaseStudy, n_cores: int = 1):
        """This method runs a study.
        """
        logger.info(f"Running study {study.name}")
        pass

    @abstractmethod
    def _run_study_docker(self,
                          study: BaseStudy,
                          docker_image: str,
                          n_cores: int = 1,
                          ):
        """This method runs a study using docker.
        """
        logger.info(f"Running study {study.name} using docker image {docker_image}")


    def run_study(self,
                  study: BaseStudy,
                  n_cores: int = 1,
                  docker_image: str = None,
                  ):
        """This method runs a study.
        """
        logger.info(f"Running study {study.name}")
        # Create the study files
        study.pre_run()
        # Run the study
        if docker_image is None:
            self._run_study(study, n_cores=n_cores)
        else:
            self._run_study_docker(study, docker_image, n_cores=n_cores)
        # Post run
        study.post_run()


    @property
    def n_studies(self):
        """This method returns the number of studies.
        """
        return len(self.studies)






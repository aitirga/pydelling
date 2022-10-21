from .BaseStudy import BaseStudy
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import logging
from alive_progress import alive_bar
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseManager(ABC):
    """This class is the base class for all simulation managers. A simulation manager should be able to:
    - Manage study [BaseStudy] objects
    - Run simulations on a given software
    - Read the output status of the simulation
    """
    def __init__(self):
        self.studies: Dict[str, BaseStudy] = {}

    def add_study(self, study: BaseStudy):
        """This method adds a study to the manager.
        """
        assert isinstance(study, BaseStudy), f"Study must be a object from a class inherited from BaseStudy, not {type(study)}"
        self.studies[study.name] = study

    def run(self):
        """This method runs all the studies.
        """
        logger.info(f"Running {self.n_studies} studies")
        for study in tqdm(self.studies.values(), desc="Running studies", colour="white"):
            self.run_study(study.name)

    @abstractmethod
    def _run_study(self, study_name: str, n_cores: int = 1):
        """This method runs a study.
        """
        logger.info(f"Running study {study_name}")
        pass

    @abstractmethod
    def _get_study_status(self, study_name: str):
        """This method returns the status of a study.
        """
        pass

    def run_study(self, study_name: str, n_cores: int = 1):
        """This method runs a study.
        """
        logger.info(f"Running study {study_name}")
        self._run_study(study_name, n_cores=n_cores)

    def get_study_status(self, study_name: str):
        """This method returns the status of a study.
        """
        return self._get_study_status(study_name)

    @property
    def n_studies(self):
        """This method returns the number of studies.
        """
        return len(self.studies)







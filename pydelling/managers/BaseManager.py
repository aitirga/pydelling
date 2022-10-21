from .BaseStudy import BaseStudy
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class BaseManager(ABC):
    """This class is the base class for all simulation managers. A simulation manager should be able to:
    - Manage study [BaseStudy] objects
    - Run simulations on a given software
    - Read the output status of the simulation
    """
    def __init__(self):
        self.studies: Dict[str, BaseStudy]  = {}

    def add_study(self, study: BaseStudy):
        """This method adds a study to the manager.
        """
        self.studies[study.name] = study

    def run(self):
        """This method runs all the studies.
        """
        for study in self.studies.values():
            self.run_study(study.name)

    @abstractmethod
    def run_study(self, study_name: str):
        """This method runs a study.
        """
        pass

    @abstractmethod
    def get_study_status(self, study_name: str):
        """This method returns the status of a study.
        """
        pass







from .BaseStudy import BaseStudy
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import logging
from alive_progress import alive_bar
from tqdm import tqdm
from pydelling.utils import create_results_folder
from pydelling.managers.ssh import BaseSsh
from docker import DockerClient
from pathlib import Path
import subprocess


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
        self.is_dummy = False
        self.ssh: BaseSsh = None
        self.password = None


    def add_study(self, study: BaseStudy):
        """This method adds a study to the manager.
        """
        assert isinstance(study, BaseStudy), f"Study must be a object from a class inherited from BaseStudy, not {type(study)}"
        self.studies[study.name] = study

    def run(self,
            studies_folder: str = './studies',
            n_cores: int = 1,
            docker_image: str = None,
            dummy: bool = False,
            start_from: int = None,
            petsc_dir: str = '/opt/pflotran-dev/petsc',
            petsc_arch: str = 'arch-linux-c-opt',
            pre_commands: List[str] = None,
            run_on_jureca: bool = False,
            user: str = None,
            project_name: str = None,
            pkey_path: str = None,
            wallclock_limit: float = 23,
            shell_script: str = None,
            **kwargs,
            ):
        """This method runs all the studies.
        """
        self.is_dummy = dummy
        shell_script = Path(shell_script).absolute() if shell_script is not None else None
        # Initialize callbacks
        for study in self.studies.values():
            study.initialize_callbacks(self)

        self.results_folder = create_results_folder(studies_folder)
        self.studies_folder_name = Path(studies_folder).name
        # self.generate_run_files(studies_folder=studies_folder)
        # Run the precommands if any
        if pre_commands is not None:
            for command in pre_commands:
                logger.info(f"Running precommand: {command}")
                subprocess.run(command, shell=True)
        for study in tqdm(self.studies.values(), desc="Running studies", colour="white"):
            study: BaseStudy
            if start_from is not None:
                if study.idx < start_from - 1:
                    logger.info(
                        f"Skipping study {study.name} (idx: {study.idx}) because start_from is set to {start_from}")
                    self.run_study(study,
                                   docker_image=docker_image,
                                   n_cores=n_cores,
                                   dummy=True,
                                   petsc_dir=petsc_dir,
                                   petsc_arch=petsc_arch,
                                   run_on_jureca=run_on_jureca,
                                   user=user,
                                   project_name=project_name,
                                   pkey_path=pkey_path,
                                   wallclock_limit=wallclock_limit,
                                   shell_script_path=shell_script,
                                   **kwargs)
                    continue
            self.run_study(study,
                           docker_image=docker_image,
                           n_cores=n_cores,
                           dummy=dummy,
                           petsc_dir=petsc_dir,
                           petsc_arch=petsc_arch,
                           run_on_jureca=run_on_jureca,
                           user=user,
                           project_name=project_name,
                           pkey_path=pkey_path,
                           wallclock_limit=wallclock_limit,
                           shell_script_path=shell_script,
                           **kwargs)

    def generate_run_files(self, studies_folder: str = './studies'):
        """This method generates the run files for all the studies.
        """
        for study in self.studies.values():
            study: BaseStudy
            study.to_file(self.results_folder / study.name)

    @abstractmethod
    def _run_study(self, study: BaseStudy, n_cores: int = 1, **kwargs):
        """This method runs a study.
        """
        logger.info(f"Running study {study.name}")
        return NotImplementedError("This method must be implemented in the child class")

    @abstractmethod
    def _run_study_docker(self,
                          study: BaseStudy,
                          docker_image: str,
                          n_cores: int = 1,
                          **kwargs,
                          ):
        """This method runs a study using docker.
        """
        logger.info(f"Running study {study.name} using docker image {docker_image}")
        return NotImplementedError("This method must be implemented in the child class")

    def _run_study_jureca(self,
                            study: BaseStudy,
                            user: str,
                            project_name: str,
                            pkey_path: str,
                            n_cores: int = 1,
                            wallclock_limit: float = None,
                            shell_script: str = None,
                            **kwargs,
                            ):
        """This method runs a study in JURECA."""
        logger.info(f"Running study {study.name} in JURECA")
        return NotImplementedError("This method must be implemented in the child class")

    def run_study(self,
                  study: BaseStudy,
                  n_cores: int = 1,
                  docker_image: str = None,
                  run_on_jureca: bool = False,
                  dummy: bool = False,
                  user: str = None,
                  project_name: str = None,
                  pkey_path: str = None,
                  wallclock_limit: float = 23,
                  shell_script: str = None,
                  **kwargs,
                  ):
        """This method runs a study.
        """
        logger.info(f"Running study {study.name}")
        # Create the study files
        study.output_folder = self.results_folder / study.name
        if dummy:
            logger.info("Dummy run, not running the study")
        else:
            for callback in study.callbacks:
                if callback.kind == 'pre':
                    callback.run()
            study.to_file(self.results_folder / study.name)
            # Run the study
            if run_on_jureca:
                self._run_study_jureca(study,
                                       n_cores=n_cores,
                                       user=user,
                                       project_name=project_name,
                                       pkey_path=pkey_path,
                                       wallclock_limit=wallclock_limit,
                                       shell_script=shell_script,
                                       **kwargs)
            elif docker_image is not None:
                self._run_study_docker(study, docker_image, n_cores=n_cores, **kwargs)
            else:
                self._run_study(study, n_cores=n_cores, **kwargs)
            for callback in study.callbacks:
                if callback.kind == 'post':
                    callback.run()
            study.post_run()


    @property
    def n_studies(self):
        """This method returns the number of studies.
        """
        return len(self.studies)

    def merge_studies(self):
        """This methods merges the studies generated by the manager.
        """
        pass






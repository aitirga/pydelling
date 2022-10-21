from .BaseManager import BaseManager
import subprocess

class PflotranManager(BaseManager):
    def _get_study_status(self, study_id: int):
        """This method returns the status of a study.
        """
        pass

    def _run_study(self,
                   study_name: str,
                   n_cores: int = 1):
        """This method runs a study.
        """
        study = self.studies[study_name]
        if n_cores == 1:
            # Run the study in serial
            subprocess.run(['pflotran', '-pflotranin', study.input_file.name])
        else:
            # Run the study in parallel
            subprocess.run(['$PETSC_DIR/$PETSC_ARCH/bin/mpirun', '-np', str(n_cores), 'pflotran', '-pflotranin', study.input_file.name])







from .BaseManager import BaseManager
from . import PflotranStudy
import subprocess
import docker
from docker.models.containers import Container
from io import BytesIO
from pathlib import Path
from pydelling.utils import create_results_folder
from pydelling.managers.PflotranPostprocessing import PflotranPostprocessing
import logging
import os

logger = logging.getLogger(__name__)
class PflotranManager(BaseManager):
    def _get_study_status(self, study_id: int):
        """This method returns the status of a study.
        """
        pass

    def _run_study(self,
                   study: PflotranStudy,
                   n_cores: int = 1):
        """This method runs a study.
        """
        # Change the working directory to the study folder
        os.chdir(study.output_folder)
        # test comment
        if n_cores == 1:
            # Run the study in serial
            subprocess.run(['pflotran', '-pflotranin', study.input_file.name])
        else:
            # Run the study in parallel
            subprocess.run(['$PETSC_DIR/$PETSC_ARCH/bin/mpirun', '-np', str(n_cores), 'pflotran', '-pflotranin', study.input_file.name])

    def _run_study_docker(self,
                          study: PflotranStudy,
                          docker_image: str,
                          n_cores: int = 1,
                          ):
        """This method runs a study using docker.
        """
        docker_client = docker.from_env()
            # Run the study in serial

        try:
            docker_client.api.start(self.manager_name)
            container = docker_client.containers.get(self.manager_name)
        except:
            docker_client.api.create_container(
                name=self.manager_name,
                image=docker_image,
                command='/bin/sh',
                detach=True,
                tty=True,
                volumes=[f'/home/pflotran/'],
                host_config=docker_client.api.create_host_config(
                    binds={
                        Path().cwd() / f'studies/{study.name}': {
                            'bind': f'/home/pflotran/{study.name}',
                            'mode': 'rw',
                        }
                    })
            )
            docker_client.api.start(container=self.manager_name)
            container = docker_client.containers.get(self.manager_name)

        # Compress study folder to a tar file
        bytes_io = BytesIO()
        import tarfile
        pw_tar = tarfile.TarFile(fileobj=bytes_io, mode='w')
        pw_tar.add(study.output_folder, arcname=study.output_folder.name)
        pw_tar.close()
        bytes_io.seek(0)
        # volume = docker_client.volumes.create(name=study.output_folder.name, driver='local', driver_opts={'type': 'none', 'device': study.output_folder, 'o': 'bind'})
        # Copy the tar file to the container
        container.exec_run(cmd="bash -c 'cd /home/pflotran'")
        container.put_archive('/home/pflotran', bytes_io)
        container.exec_run(cmd="bash -c 'ls'")
        if n_cores == 1:
            container.exec_run(cmd=f"sudo bash -c 'export PATH=$PATH:/home/pflotran/pflotran/src/pflotran &&"
                                   f" cd /home/pflotran/{study.output_folder.name} && "
                                   f"pflotran -pflotranin {study.input_file_name}'")
        else:
            result = container.exec_run(cmd=f"sudo bash -c 'export PATH=$PATH:/home/pflotran/pflotran/src/pflotran && "
                                   f"export PETSC_DIR=/home/pflotran/petsc && "
                                   f"export PETSC_ARCH=docker-petsc && "
                                   f"cd /home/pflotran/{study.output_folder.name} && "
                                   f"$PETSC_DIR/$PETSC_ARCH/bin/mpiexec -np {n_cores} pflotran -pflotranin {study.input_file_name}'")
        container.stop()
        container.remove()

    def merge_results(self, move=False, postprocess=True):
        """This method merges the results of all the studies.
        """
        self.run(dummy=True)
        import shutil
        # results_folder = create_results_folder(list(self.studies.values())[0].output_folder)
        self.merged_folder = create_results_folder(self.results_folder / 'merged_results')
        for study in self.studies.values():
            hdf5_files = list(study.output_folder.glob('*.h5'))
            for file in hdf5_files:
                if 'restart' not in file.name:
                    if not move:
                        shutil.copy(file, self.results_folder / 'merged_results')
                    else:
                        shutil.move(file, self.results_folder / 'merged_results')

        # Copy -domain.h5 file


        if postprocess:
            domain_folder = list(self.studies.values())[0].output_folder / 'input_files'
            domain_file = list(domain_folder.glob('*-domain.h5'))[0]
            shutil.copy(domain_file, self.results_folder / 'merged_results')
            logger.info('Postprocessing results')
            pflotran_postprocesser = PflotranPostprocessing()
            # Change the working directory to the results folder
            import os
            os.chdir(self.results_folder / 'merged_results')
            pflotran_postprocesser.run()
            # Return to the original working directory

















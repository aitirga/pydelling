from .BaseManager import BaseManager
from . import PflotranStudy
import subprocess
import docker
from docker.models.containers import Container
from io import BytesIO
from pathlib import Path
from pydelling.utils import create_results_folder
from pydelling.managers.PflotranPostprocessing import PflotranPostprocessing
from pydelling.managers.ssh import JurecaSsh
from pydelling.managers.status import PflotranStatus
import logging
import os
import getpass
import os

logger = logging.getLogger(__name__)
class PflotranManager(BaseManager):
    def _get_study_status(self, study_id: int):
        """This method returns the status of a study.
        """
        pass

    def _run_study(self,
                   study: PflotranStudy,
                   n_cores: int = 1,
                   petsc_dir: str = '/opt/pflotran-dev/petsc',
                   petsc_arch: str = 'arch-linux-c-opt',
                   **kwargs,
                   ):
        """This method runs a study.
        """
        if n_cores == 1:
            # Run the study in serial
            subprocess.run(['pflotran', '-pflotranin', study.input_file_name], cwd=study.output_folder.absolute())
        else:
            # Run the study in parallel
            os.environ['PETSC_DIR'] = petsc_dir
            os.environ['PETSC_ARCH'] = petsc_arch
            subprocess.call([f'$PETSC_DIR/$PETSC_ARCH/bin/mpirun -np {n_cores} pflotran -pflotranin {study.input_file_name}'],
                           cwd=study.output_folder.absolute(),
                           shell=True
                           )

    def _run_study_docker(self,
                          study: PflotranStudy,
                          docker_image: str,
                          n_cores: int = 1,
                          **kwargs,
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

    def _run_study_jureca(self,
                            study: PflotranStudy,
                            user,
                            project_name,
                            pkey_path,
                            n_cores: int = 1,
                            wallclock_limit: str = None,
                            shell_script_path: str = None,
                            **kwargs,
                            ):
        """This method runs a study on JURECA"""
        if self.password is None:
            self.password = getpass.getpass(prompt='Password: ', stream=None)

        self.ssh = JurecaSsh(user=user,
                             project_name=project_name,
                             pkey_path=pkey_path,
                             password=self.password,
                             )
        # Create a folder for the set of studies
        self.ssh.cd_project()
        self.ssh.mkdir(self.studies_folder_name)
        # Create a folder for the study, delete it if it already exists
        self.ssh.cd(self.studies_folder_name)
        if study.name in self.ssh.ls:
            self.ssh.rmdir(study.name)
        self.ssh.mkdir(study.name)
        # Copy the study folder to the server
        self.ssh.cpdir(study.output_folder, study.name)
        # Copy the shell_script to the server
        if shell_script_path is not None:
            # Read the shell script and replace the study name where {input_name} is found\
            with open(shell_script_path, 'r') as f:
                shell_script = f.read()
            shell_script = shell_script.replace('{input_name}', study.input_file_name)
            # Write to temporary file
            with open('temp.sh', 'w') as f:
                f.write(shell_script)
            # Delete the temporary file
            self.ssh.cp('temp.sh', f"{study.name}/{Path(shell_script_path).name}")
            os.remove('temp.sh')
            # Run the shell script
            self.ssh.send_job(f"{study.name}/{Path(shell_script_path).name}")
            # Get the highest job id
            job_id = max(self.ssh.user_queue['JOBID'])
            # pflotran_status = PflotranStatus()
            self.ssh.wait_for_job(job_id)
            # Copy the results back to the local machine
            # Detect files ending with .h5
            dir_list = self.ssh.ls
            dir_list = [file for file in dir_list if file.endswith('.h5')]
            # Add the files containing the job id to the list
            dir_list += [file for file in self.ssh.ls if str(job_id) in file]
            for file in dir_list:
                self.ssh.get(f"{file}", study.output_folder/file)

        else:
            raise ValueError('shell_script should be provided with the details of the job submission.')









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
            os.chdir(self.results_folder / 'merged_results')
            pflotran_postprocesser.run()
            # Return to the original working directory

















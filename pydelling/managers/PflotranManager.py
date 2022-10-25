from .BaseManager import BaseManager
import subprocess
import docker
from docker.models.containers import Container
from io import BytesIO
from pathlib import Path

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

    def _run_study_docker(self,
                          study_name: str,
                          docker_image: str,
                          n_cores: int = 1,
                          ):
        """This method runs a study using docker.
        """
        docker_client = docker.from_env()
        study = self.studies[study_name]
            # Run the study in serial
            # Add current folder to the container
        container = docker_client.api.create_container(
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
        docker_client.api.start(container=container['Id'])
        container = docker_client.containers.get(container['Id'])
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
            print(result)
        container.stop()
        container.remove()








from pydelling.managers.ssh import BaseSsh
import paramiko
import getpass
import logging
import pandas as pd
from pathlib import Path
import rich
from pydelling.managers.status import BaseStatus
logger = logging.getLogger(__name__)


class JurecaSsh(BaseSsh):
    hostname = 'jureca.fz-juelich.de'
    def __init__(self,
                 user,
                 pkey_path,
                 project_name,
                 password=None,
                 ):
        """
        Connects to the remote server.
        Args:
            user: username
            pkey_path: path to the private key
        """
        super().__init__(user, pkey_path, password)
        self.project_name = project_name
        self.current_job_id = None

    def connect(self):
        # Ask for password using a hidden input
        if self.password is None:
            self.password = getpass.getpass(prompt='Password: ', stream=None)
        self.pkey = paramiko.ed25519key.Ed25519Key.from_private_key_file(self.pkey_path, password=self.password)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=self.hostname,
            username=self.user,
            password=self.password,
            pkey=self.pkey,
        )
        self.sftp = paramiko.SFTPClient.from_transport(self.client.get_transport())


    @property
    def general_queue(self) -> pd.DataFrame:
        """
        Returns the queue status in the remote server.
        Returns: queue status
        """
        queue_string = self.run_command("squeue")
        lines = queue_string.split('\n')
        lines = [line.split() for line in lines]
        # Take only the length of the first line
        lines = [line[:len(lines[0])] for line in lines]
        return pd.DataFrame(lines[1:], columns=lines[0])

    @property
    def user_queue(self) -> pd.DataFrame:
        """
        Returns the queue status in the remote server.
        Returns: queue status
        """
        queue_string = self.run_command("squeue -u aitor1")
        lines = queue_string.split('\n')
        lines = [line.split() for line in lines]
        # Take only the length of the first line
        lines = [line[:len(lines[0])] for line in lines]
        return pd.DataFrame(lines[1:], columns=lines[0])

    def cd_project(self):
        """
        Changes the current working directory to the project directory in the remote server.
        """
        self.cd(f'/p/scratch/cjiek63/{self.project_name}')

    def send_job(self, job_path):
        """
        Sends a job to the remote server.
        Args:
            job_path: path to the job file
        """
        job_path = Path(job_path)
        logger.info(f'Sending job {job_path} to the remote server')
        self.cd(job_path.parent)
        self.run_command(f"cd {self.pwd} && sbatch {job_path.name}")

    def wait_for_job(self,
                     job_id,
                     check_interval=3,
                     running_object_status: BaseStatus = None,
                     ):
        """
        Waits for a job to finish in the remote server.
        Args:
            job_id: id of the job
            check_interval: time in seconds between checks

        Returns:

        """
        import time
        from rich.progress import Progress

        logger.info(f'Waiting for job {job_id} to finish in the remote server')
        is_finished = False
        # Add a static rich status information bar (not a Progress object) that can be updated dynamically
        with Progress() as status_bar:
            task_wait = status_bar.add_task(f"Waiting the job {job_id} to start", total=None)
            task_start = status_bar.add_task(f"Starting the job {job_id}", total=None, visible=False)
            task_cancel = status_bar.add_task(f"Canceling the job {job_id}", total=None, visible=False)
            task_run = status_bar.add_task(f"Running the job {job_id}", total=None if running_object_status is None else 100, visible=False)
            assert_finish_counter = 0

            while not is_finished:
                job_status = self.get_job_status(job_id)
                if job_status == 'CF':
                    assert_finish_counter = 0
                    status_bar.update(task_wait, visible=False)
                    status_bar.update(task_cancel, visible=False)
                    status_bar.update(task_run, visible=False)
                    status_bar.update(task_start, visible=True)
                elif job_status == 'PD':
                    assert_finish_counter = 0
                    status_bar.update(task_wait, visible=True)
                    status_bar.update(task_cancel, visible=False)
                    status_bar.update(task_run, visible=False)
                    status_bar.update(task_start, visible=False)
                elif job_status == 'R':
                    assert_finish_counter = 0
                    status_bar.update(task_wait, visible=False)
                    status_bar.update(task_cancel, visible=False)
                    status_bar.update(task_run, visible=True)
                    status_bar.update(task_start, visible=False)
                    if running_object_status is not None:
                        running_object_status.read_status_file()
                        status_bar.update(task_run, completed=running_object_status.progress)
                elif job_status == 'CG':
                    status_bar.update(task_wait, visible=False)
                    status_bar.update(task_cancel, visible=True)
                    status_bar.update(task_run, visible=False)
                    status_bar.update(task_start, visible=False)
                elif job_status == None:
                    assert_finish_counter += 1
                    if assert_finish_counter > 3:
                        status_bar.remove_task(task_wait)
                        status_bar.remove_task(task_cancel)
                        status_bar.remove_task(task_run)
                        status_bar.remove_task(task_start)
                        is_finished = True

                time.sleep(check_interval)


    def cancel_job(self, job_id):
        """
        Cancels a job in the remote server.
        Args:
            job_id: id of the job
        """
        logger.info(f'Cancelling job {job_id} in the remote server')
        self.run_command(f"scancel {job_id}")

    def cancel_all_jobs(self):
        """
        Cancels all jobs in the remote server.
        """
        get_user_jobs = self.user_queue['JOBID']
        for job in get_user_jobs:
            self.cancel_job(job)

    def get_job_status(self, job_id):
        """
        Returns the status of a job in the remote server.
        Args:
            job_id: id of the job

        Returns: status of the job
        """
        try:
            return self.general_queue[self.general_queue['JOBID'] == job_id]['ST'].values[0]
        except:
            return None

    @property
    def user_job_ids(self):
        """
        Returns the jobs of the user in the remote server.
        Returns: jobs of the user
        """
        return self.user_queue['JOBID'].values





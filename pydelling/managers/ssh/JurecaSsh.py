from pydelling.managers.ssh import BaseSsh
import paramiko
import getpass
import logging
import pandas as pd
from pathlib import Path
logger = logging.getLogger(__name__)


class JurecaSsh(BaseSsh):
    hostname = 'jureca.fz-juelich.de'
    def __init__(self, user,
                 pkey_path,
                 project_name,
                 ):
        """
        Connects to the remote server.
        Args:
            user: username
            pkey_path: path to the private key
        """
        super().__init__(user, pkey_path)
        self.project_name = project_name

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





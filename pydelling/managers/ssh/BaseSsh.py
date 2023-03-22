from abc import ABC, abstractmethod
import pandas as pd
import paramiko
import logging
logger = logging.getLogger(__name__)
from pathlib import Path

class BaseSsh(ABC):
    client: paramiko.SSHClient
    sftp: paramiko.SFTPClient
    def __init__(self, user,
                 pkey_path,
                 password=None):
        """
        Connects to the remote server.
        Args:
            user: username
            pkey_path: path to the private key
        """
        self.pkey_path = pkey_path
        self.user = user
        self.password = password
        self.connect()

    @abstractmethod
    def connect(self):
        pass

    def run_command(self, command):
        """
        Runs a ssh command in the remote server.
        Args:
            command: command to run

        Returns: stdout

        """
        stdin, stdout, stderr = self.client.exec_command(command)
        return stdout.read().decode('utf-8').strip()

    @property
    def pwd(self):
        """
        Returns the current working directory in the remote server.
        Returns: current working directory
        """
        return self.sftp.getcwd()

    def cd(self, path):
        """
        Changes the current working directory in the remote server.
        Args:
            path: path to change to
            """
        path = str(path)
        self.sftp.chdir(path)
        logger.info(f'Changed directory to {path}')

    def mkdir(self, path):
        """
        Creates a directory in the remote server.
        Args:
            path: path to create
        """
        path = Path(path)
        if path.name in self.ls:
            logger.info(f'Directory {path} already exists')
            return
        else:
            self.sftp.mkdir(str(path))
            logger.info(f'Created directory {path}')

    def rm(self, path):
        """
        Removes a file in the remote server.
        Args:
            path: file to remove

        Returns:

        """
        path = str(path)
        self.sftp.remove(path)
        logger.info(f'Removed file {path}')

    def rmdir(self,
              path,
              ):
        """
        Removes a directory in the remote server.

        Args:
            path: directory to remove

        Returns:

        """
        path = Path(path)
        for file in self.sftp.listdir(str(path)):
            try:
                self.rm(str(path / file))
            except IOError:
                self.rmdir(str(path / file))
        self.sftp.rmdir(str(path))
        logger.info(f'Removed directory {path}')

    def cp(self, src, dst):
        """
        Copies a file from the local machine to the remote server.
        Args:
            src: source file
            dst: destination file
        """
        self.sftp.put(src, dst)
        logger.info(f'Copied {src} to {dst}')

    def cpdir(self, src, dst):
        """
        Copies a directory from the local machine to the remote server.
        Args:
            src: source directory
            dst: destination directory
        """
        src = Path(src)
        dst = Path(dst)

        self.mkdir(str(dst))
        for file in src.iterdir():
            if file.is_dir():
                self.cpdir(file, str(dst / file.name))
            else:
                self.cp(file, str(dst / file.name))

    def get(self, src, dst):
        """
        Copies a file from the remote server to the local machine.
        Args:
            src: source file
            dst: destination file
        """
        self.sftp.get(src, dst)
        logger.info(f'Copied {src} to {dst}')

    def getdir(self, src, dst):
        """
        Copies a directory from the remote server to the local machine.
        Args:
            src: source directory
            dst: destination directory
        """
        src = Path(src)
        dst = Path(dst)

        dst.mkdir(parents=True, exist_ok=True)
        self.cd(src)
        for file in self.ls:
            self.get(file, dst / Path(file).name)


    @property
    def ls(self):
        """
        Returns the content of the current working directory in the remote server.
        Returns: content of the current working directory
        """
        return self.sftp.listdir()


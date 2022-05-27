from abc import ABC
from pathlib import Path
import inspect
import subprocess


class WebAppRunner(ABC):
    """This is the base class used to build other webapps."""
    def __init__(self):
        self.source_script_name = None
        if 'threading' not in inspect.stack()[-1][1]:
            self.source_script_name = inspect.stack()[-1][1]

    def construct(self):
        """This is the main method that builds the webapp."""
        pass

    def run(self):
        """This is the main method that runs the webapp."""
        current_executer = inspect.stack()
        if 'threading' not in current_executer[-1][1]:
            subprocess.run(["streamlit", "run", self.source_script_name])
        else:
            self.construct()







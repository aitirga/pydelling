from pathlib import Path
from jinja2 import Template

import logging

logger = logging.getLogger(__name__)


class BaseManager:
    """This class contains the core methods for all the other child Manager classes.
    A manager is supposed to control the simulations run on a given software automatically.

    The manager should be able to:
    - Create the input files for the software
    - Modify the input files for the software
    - Run the software
    - Read the output status of the simnulation

    The idea to process the input files is the following:
    - Read the raw input file
    - The user can add variables to change, or can specify directly on the input file using {{var}} notation (jinja2)
    - The input file is rendered using jinja2 for each specific case
    """

    def __init__(self, input_file: str):
        """This method initializes the manager.
        """
        logger.info(f"Initializing {self.__class__.__name__} manager")
        self.input_file = Path(input_file)
        self.settings = {}
        self.jinja_settings = {}
        # Step 1: read and set the input file
        self.raw_text = self.input_file.read_text()
        self.aux_files = {}

    def _replace_with_jinja_variable(self, var, var_name=None, value=None):
        """This method replaces a variable in the raw text with a jinja variable.
        """
        logger.debug(f"Replacing {var} with jinja variable")
        if var_name is None:
            var_name = var
        self.raw_text = self.raw_text.replace(var, f"{{{{ {var_name} }}}}")
        self.jinja_settings[var_name] = {"value": value}

    def render(self, **kwargs):
        """This method renders the input file using jinja2.
        """
        logger.info(f"Rendering input file {self.input_file}")
        template = Template(self.raw_text)
        return template.render(**kwargs)


    def add_auxiliary_file(self, file_path: str):
        """This method adds an auxiliary file to the manager.
        """
        logger.info(f"Adding auxiliary file {file_path}")
        file_path = Path(file_path)
        self.aux_files[file_path.name] = file_path

    def add_input_file(self, file_path: str):
        """This method adds an auxiliary file to the manager.
        """
        logger.info(f"Adding input file {file_path}")
        file_path = Path(file_path)
        self.aux_files[file_path.name] = file_path

    def to_file(self,
                output_folder: str='result',
                output_file: str=None,
                auxiliary_folder: str=None,
                **kwargs):
        """This method renders the input file and saves it to a file.
        """
        logger.info(f"Saving input files to {output_folder}")
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / (output_file or self.input_file.name)
        output_file.write_text(self.render(**kwargs))

        # Copy the auxiliary files
        auxiliary_folder = auxiliary_folder if auxiliary_folder is not None else 'input_files'
        auxiliary_folder = output_folder / auxiliary_folder
        auxiliary_folder.mkdir(exist_ok=True)
        for aux_file in self.aux_files:
            self.aux_files[aux_file].copy(auxiliary_folder / aux_file)








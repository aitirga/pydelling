from pathlib import Path
from jinja2 import Template
import shutil
from pydelling.utils import UnitConverter

import logging

logger = logging.getLogger(__name__)


class BaseStudy(UnitConverter):
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
    count = 0

    def __init__(self, input_file: str):
        """This method initializes the manager.
        """
        self.idx = self.__class__.count
        self.__class__.count += 1
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

    def _find_tags(self, tag: str, ignore_case: bool = True):
        """This method finds the line index of the tag in the raw text.
        """
        logger.debug(f"Finding tag '{tag}'")
        lines = self.raw_text.splitlines()
        if ignore_case:
            lines = [line.lower() for line in lines]
            tag = tag.lower()
        # Delete lines starting with #
        # lines = [line for line in lines if not line.startswith('#')]
        line_idx = [i for i, line in enumerate(lines) if tag in line]
        # Clean for comments
        line_idx = [i for i in line_idx if not lines[i].startswith('#')]
        return line_idx

    def _find_tags_in_subset(self, tags: list, subset: list, ignore_case: bool = True):
        """This method finds the line index of the tag in the raw text.
        """
        logger.debug(f"Finding tags '{tags}' in subset")
        if ignore_case:
            subset = [line.lower() for line in subset]
            tags = [tag.lower() for tag in tags]
        return [i for i, line in enumerate(subset) if any(tag in line for tag in tags)]

    def _get_line(self, line_index: int):
        """This method returns the line of the raw text.
        """
        return self.raw_text.splitlines()[line_index]

    def _add_line(self, line_index: int, new_line: list, sep: str = ' '):
        """This method adds a line in the raw text.
        """
        logger.debug(f"Adding line {line_index} with {new_line}")
        lines = self.raw_text.splitlines()
        lines.insert(line_index, sep.join(new_line))
        self.raw_text = '\n'.join(lines)

    def _get_nth_previous_line(self, line_index: int, n: int = 1):
        """This method returns the nth previous line of the raw text.
        """
        return self.raw_text.splitlines()[line_index - n]

    def _get_nth_next_line(self, line_index: int, n: int = 1):
        """This method returns the nth next line of the raw text.
        """
        return self.raw_text.splitlines()[line_index + n]

    def _replace_line(self, line_index: int, new_line: list, sep: str = ' '):
        """This method replaces a line in the raw text.
        """
        logger.debug(f"Replacing line {line_index} with {new_line}")
        lines = self.raw_text.splitlines()
        lines[line_index] = sep.join(new_line)
        self.raw_text = '\n'.join(lines)



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
                output_folder: str=None,
                output_file: str=None,
                auxiliary_folder: str=None,
                **kwargs):
        """This method renders the input file and saves it to a file.
        """
        output_folder = output_folder if output_folder is not None else f'case-{BaseStudy.count}'
        logger.info(f"Saving input files to {output_folder}")
        BaseStudy.count += 1
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / (output_file or self.input_file.name)
        output_file.write_text(self.render(**kwargs))

        # Copy the auxiliary files
        if len(self.aux_files) > 0:
            auxiliary_folder = auxiliary_folder if auxiliary_folder is not None else 'input_files'
            auxiliary_folder = output_folder / auxiliary_folder
            auxiliary_folder.mkdir(exist_ok=True)
            for aux_file in self.aux_files:
                shutil.copy(self.aux_files[aux_file], auxiliary_folder / aux_file)

    def __repr__(self):
        return f"{self.__class__.__name__} (input template: {self.input_file.name})"

    # Properties
    @property
    def name(self):
        return f"{self.__class__.__name__}-{self.__class__.count}"









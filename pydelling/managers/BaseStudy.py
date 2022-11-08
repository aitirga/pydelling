from __future__ import annotations
from pathlib import Path
from jinja2 import Template
import shutil
from pydelling.utils import UnitConverter
from typing import Callable

from typing import TYPE_CHECKING, List, Union
if TYPE_CHECKING:
    from pydelling.managers import BaseCallback, BaseManager

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

    def __init__(self,
                 input_file: str,
                 study_name: str = None,
                 is_independent: bool = False,
                 input_file_name: str = None,
                 ):
        """This method initializes the manager.
        """
        if not is_independent:
            self.idx = self.__class__.count
            self.__class__.count += 1
            self.is_independent = False
        else:
            self.idx = None
            self.is_independent = True
        self.name = study_name if study_name is not None else f"{self.__class__.__name__}-{self.idx + 1}"
        self.input_file_name = input_file_name if input_file_name is not None else Path(input_file).name
        logger.info(f"Initializing {self.__class__.__name__} study")
        self.input_file = Path(input_file)
        self.settings = {}
        self.jinja_settings = {}
        # Step 1: read and set the input file
        self.raw_text = self.input_file.read_text()
        self.aux_files = {}
        self.output_folder = None
        self._callbacks: List[Callable] = []
        self.callbacks: List[BaseCallback] = []

    def pre_run(self):
        """This method is executed before the run.
        """
        pass

    def post_run(self):
        """This method is executed after the run.
        """
        pass

    def _replace_with_jinja_variable(self, var, var_name=None, value=None):
        """This method replaces a variable in the raw text with a jinja variable.
        """
        logger.debug(f"Replacing {var} with jinja variable")
        if var_name is None:
            var_name = var
        self.raw_text = self.raw_text.replace(var, f"{{{{ {var_name} }}}}")
        self.jinja_settings[var_name] = {"value": value}

    def _find_tags(self, tag: str, ignore_case: bool = True, equal: bool = True):
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
        # if equal:
        #     lines_strip = [line.strip().split()[0] for line in lines]
        #     print(lines_strip)
        #     line_idx = [i for i in line_idx if lines[i].strip().strip()[0] == tag]
        #     print(line_idx)
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
        logger.info(f"Rendering input file {self.input_file_name}")
        template = Template(self.raw_text)
        return template.render(**kwargs)


    def add_auxiliary_file(self, file_path: str):
        """This method adds an auxiliary file to the manager.
        """
        logger.info(f"Adding auxiliary file {file_path}")
        file_path = Path(file_path)
        self.aux_files[file_path.name] = file_path

    def add_input_file(self, file_path: Union[Path, str]):
        """This method adds an auxiliary file to the manager.
        """
        logger.info(f"Adding input file {file_path}")
        file_path = Path(file_path)
        self.aux_files[file_path.name] = file_path

    def add_input_folder(self, folder_path: str):
        """This method adds an auxiliary folder to the manager.
        """
        folder_path = Path(folder_path)
        for file in folder_path.glob('*'):
            self.aux_files[file.name] = file
        logger.info(f"Adding input folder {folder_path} with {len(self.aux_files)} files")

    def add_callback(self, callback: Callable, kind: str = 'pre', **kwargs):
        """This method adds a callback to the manager.
        """
        kwargs['kind'] = kind
        self._callbacks.append(lambda manager: callback(manager, self, **kwargs))

    def initialize_callbacks(self, manager: BaseManager):
        """This method initializes the callbacks.
        """
        temp_callbacks = []
        for callback in self._callbacks:
            temp_callbacks.append(callback(manager))
        self.callbacks = temp_callbacks


    def to_file(self,
                output_folder: str=None,
                output_file: str=None,
                auxiliary_folder: str=None,
                **kwargs):
        """This method renders the input file and saves it to a file.
        """
        self.output_folder = output_folder if output_folder is not None else f'case-{BaseStudy.count}'
        logger.info(f"Saving input files to {output_folder}")
        BaseStudy.count += 1
        output_folder = Path(self.output_folder)
        output_folder.mkdir(exist_ok=True)
        output_file = output_folder / (output_file if output_file is not None else Path(self.input_file_name))
        output_file.write_text(self.render(**kwargs))

        # Copy the auxiliary files
        if len(self.aux_files) > 0:
            auxiliary_folder = auxiliary_folder if auxiliary_folder is not None else 'input_files'
            auxiliary_folder = output_folder / auxiliary_folder
            auxiliary_folder.mkdir(exist_ok=True)
            for aux_file in self.aux_files:
                shutil.copy(self.aux_files[aux_file], auxiliary_folder / aux_file)

    def to_tar(self):
        """This method creates a tar file with the input files. Using the gzip compression.
        """
        import tarfile
        import os
        logger.info(f"Creating tar file {self.output_folder}.tar")
        self.to_file(output_folder='temp')
        # Create tar file using gzip compression
        with tarfile.open(f"{self.output_folder}.tar", "w:gz") as tar:
            tar.add('temp', arcname=os.path.basename('temp'))
        shutil.rmtree('temp')

    def __repr__(self):
        return f"{self.__class__.__name__} (input template: {self.input_file.name})"

    # Properties
    def copy(self):
        """This method returns a copy of the object.
        """
        new_obj = self.__class__(self.input_file)
        for attr in self.__dict__:
            if attr in ['idx', 'name']:
                continue
            setattr(new_obj, attr, getattr(self, attr))
        return new_obj



    def __str__(self):
        return self.__repr__()













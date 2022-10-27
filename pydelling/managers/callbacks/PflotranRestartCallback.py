from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydelling.managers import PflotranManager, PflotranStudy

from pydelling.managers.callbacks.BaseCallback import BaseCallback
from pathlib import Path


class PflotranRestartCallback(BaseCallback):
    """Callback to restart Pflotran simulations."""
    def __init__(self, manager: PflotranManager, study: PflotranStudy, kind: str = 'post'):
        super().__init__(manager, study, 'post')

    def run(self):
        """This method should detect the hdf5 file in the original study and copy it to the next study"""
        output_files = list(self.study.output_folder.glob('*.in'))
        current_file = output_files[0]
        # rename the file
        new_file = self.study.output_folder / f"test_file.in"
        current_file.rename(new_file)
        # copy the file to the next study
        if self.study.idx < len(self.manager.studies):
            next_study = list(self.manager.studies.values())[self.study.idx + 1]
            next_study.add_input_file(new_file)


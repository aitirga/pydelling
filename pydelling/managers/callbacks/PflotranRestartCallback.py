from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydelling.managers import PflotranManager, PflotranStudy

from pydelling.managers.callbacks.BaseCallback import BaseCallback
from pathlib import Path


class PflotranRestartCallback(BaseCallback):
    """Callback to restart Pflotran simulations."""
    def __init__(self, manager: PflotranManager, study: PflotranStudy, kind: str = 'post'):
        super().__init__(manager, study, 'pre')

    def run(self):
        """This method should detect the hdf5 file in the previous study and copy it to the current study"""
        if self.study.idx > 0:
            prev_study: PflotranStudy = list(self.manager.studies.values())[self.study.idx - 1]
        else:
            return
        output_files = list(prev_study.output_folder.glob('*.h5'))
        target_file = None
        for file in output_files:
            if 'restart' in file.name:
                target_file = file
        if target_file is None:
            raise FileNotFoundError('Restart file not found')
        else:
            self.study.add_input_file(target_file)



from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pydelling.managers import PflotranManager, PflotranStudy

from pydelling.managers.callbacks.BaseCallback import BaseCallback
from pydelling.managers import PflotranPostprocessing
from pathlib import Path
import shutil
import logging

logger = logging.getLogger(__name__)


class PflotranSaveResultsCallback(BaseCallback):
    """Callback to restart Pflotran simulations."""
    move: bool = False
    postprocess: bool = False
    def __init__(self, manager: PflotranManager, study: PflotranStudy, kind: str = 'post', **kwargs):
        super().__init__(manager, study, 'post', **kwargs)

    def run(self):
        """This callback runs after the simulation is run, it creates a folder and copies (or moves) the results to it"""
        move = self.kwargs['move'] if 'move' in self.kwargs else False
        postprocess = self.kwargs['postprocess'] if 'postprocess' in self.kwargs else False

        output_files = list(self.study.output_folder.glob('*h5'))
        target_folder = Path(self.manager.results_folder / 'merged_results')
        target_folder.mkdir(exist_ok=True, parents=True)
        for file in output_files:
            if 'restart' not in file.name:
                if move:
                    # Check if the file already exists
                    if (target_folder / file.name).exists():
                        target_file = target_folder / file.name
                        target_file.unlink()
                    shutil.move(str(file), str(target_folder.absolute()))
                else:
                    shutil.copy(str(file), str(target_folder.absolute()))

        if postprocess:
            import os
            domain_folder = list(self.manager.studies.values())[0].output_folder / 'input_files'
            domain_file = list(domain_folder.glob('*-domain.h5'))[0]
            shutil.copy(domain_file, self.manager.results_folder / 'merged_results')
            old_cwd = os.getcwd()
            logger.info('Postprocessing results')
            pflotran_postprocesser = PflotranPostprocessing()
            # Change the working directory to the results folder
            os.chdir(self.manager.results_folder / 'merged_results')
            pflotran_postprocesser.run()
            os.chdir(old_cwd)





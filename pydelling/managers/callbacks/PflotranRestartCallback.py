from pydelling.managers import BaseCallback, PflotranManager, PflotranStudy


class PflotranRestartCallback(BaseCallback):
    """Callback to restart Pflotran simulations."""
    def __init__(self, manager: PflotranManager, study: PflotranStudy, kind: str = 'end'):
        super().__init__(manager, study, kind)

    def run(self):
        pass


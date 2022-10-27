from pydelling.managers import BaseManager, BaseStudy
from abc import ABC, abstractmethod


class BaseCallback(ABC):
    """Base class for callbacks."""
    def __init__(self, manager: BaseManager, study: BaseStudy, kind: str = None):
        self.manager = manager
        self.study = study
        self.kind = kind

    @abstractmethod
    def run(self):
        """This method executes the callback."""
        pass




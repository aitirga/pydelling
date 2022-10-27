from pydelling.managers import BaseManager, BaseStudy
from abc import ABC, abstractmethod


class BaseCallback(ABC):
    """Base class for callbacks."""
    def __init__(self, manager: BaseManager, study: BaseStudy, kind: str = None, **kwargs):
        self.manager = manager
        self.study = study
        self.kind = kind
        self.kwargs = kwargs

    @abstractmethod
    def run(self):
        """This method executes the callback."""
        pass




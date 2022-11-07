from pydelling.managers import BaseManager, BaseStudy
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseCallback(ABC):
    """Base class for callbacks."""
    def __init__(self, manager: BaseManager, study: BaseStudy, kind: str = None, **kwargs):
        self.manager = manager
        self.study = study
        assert kind in ['pre', 'post'], f"Kind must be 'pre' or 'post', not {kind}"
        self.kind = kind
        self.kwargs = kwargs
        self.is_run = False
        self.process_kwargs()

    @abstractmethod
    def run(self):
        """This method executes the callback."""
        self.is_run = True
        logger.info(f"Running callback {self.__class__.__name__} for study {self.study.name}")
        pass

    def process_kwargs(self):
        for kwarg_name, kwarg in self.kwargs.items():
            setattr(self, kwarg_name, kwarg)




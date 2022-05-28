import streamlit as st
from abc import ABC, abstractmethod
from pydelling.webapps import WebAppRunner


class BaseComponent(ABC):
    _value = None
    def __init__(self, webapp: WebAppRunner=None, *args, **kwargs):
        self.type = self.__class__.__name__
        self.finished = False
        if webapp:
            self.webapp = webapp
        self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Runs the component"""

    @property
    def value(self):
        """Returns the value of the component"""
        return self._value

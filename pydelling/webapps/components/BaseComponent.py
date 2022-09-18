from abc import ABC, abstractmethod

from pydelling.webapps import WebAppRunner
from translate import Translator
import streamlit as st
import extra_streamlit_components as stx
from ..BaseStreamlitUtilityClass import BaseStreamlitUtilityClass

class BaseComponent(ABC, BaseStreamlitUtilityClass):
    _value = None
    def __init__(self,
                 webapp: WebAppRunner=None,
                 lang=None,
                 translate=False,
                 key=None,
                 *args,
                 **kwargs
                 ):
        self.type = self.__class__.__name__
        self.finished = False
        self.translate = translate

        for kwarg in kwargs:
            setattr(self, kwarg, kwargs[kwarg])

        if webapp:
            self.webapp = webapp
        if lang is not None:
            self.translator = Translator(to_lang=lang)
            self.translate = True
        self.name = f'{self.type}{key if key is not None else ""}'
        self.initialize_in_session_state(key=f'{self.name}-init', value=False)
        self.run(*args, **kwargs)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Runs the component"""

    @property
    def value(self):
        """Returns the value of the component"""
        return self._value
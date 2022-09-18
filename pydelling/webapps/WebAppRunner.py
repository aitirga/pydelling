import inspect
import subprocess
from abc import ABC
import streamlit as st

class WebAppRunner(ABC):
    """This is the base class used to build other webapps."""
    def __init__(self):
        self.source_script_name = None
        if 'threading' not in inspect.stack()[-1][1]:
            self.source_script_name = inspect.stack()[-1][1]

    def construct(self):
        """This is the main method that builds the webapp."""
        pass

    def run(self):
        """This is the main method that runs the webapp."""
        current_executer = inspect.stack()
        if 'threading' not in current_executer[-1][1]:
            subprocess.run(["streamlit", "run", self.source_script_name])
        else:
            self.initialize()
            self.construct()

    def save_in_session_state(self, value, key: str):
        """This method saves the object to a session state"""
        if key not in st.session_state:
            st.session_state[key] = value
        else:
            st.session_state[key] = value

    def initialize_in_session_state(self, key: str, value=None):
        """This method initializes the session state"""
        if key not in st.session_state:
            st.session_state[key] = value

    def get_from_session_state(self, key: str):
        """This method gets the session state"""
        return st.session_state[key]

    def initialize(self):
        """This method initializes the webapp."""
        pass

    def set_to_session_state(self, key: str, value):
        """This method equals the previous method"""
        self.save_in_session_state(value, key)






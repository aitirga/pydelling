import streamlit as st
import extra_streamlit_components as stx

class BaseStreamlitUtilityClass:
    def save_in_session_state(self, key, value: object):
        """This method saves the object to a session state"""
        st.session_state[key] = value

    def initialize_in_session_state(self, key: str, value: object=None):
        """This method initializes the session state"""
        if key not in st.session_state:
            st.session_state[key] = value

    def get_from_session_state(self, key: str):
        """This method gets the session state"""
        return st.session_state[key]

    def initialize(self):
        """This method initializes the webapp."""
        pass

    def set_to_session_state(self, key: str, value: object):
        """This method equals the previous method"""
        self.save_in_session_state(key, value)

    @staticmethod
    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def get_manager():
        return stx.CookieManager()
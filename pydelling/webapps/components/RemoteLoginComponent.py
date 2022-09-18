from pydelling.webapps.components import BaseComponent
import streamlit as st
import time


class RemoteLoginComponent(BaseComponent):
    def __int__(self,
                host: str,
                host_name: str = None,
                *args,
                **kwargs
                ):
        self.initialize_in_session_state('host')
        self.initialize_in_session_state('host_name')
        self.initialize_in_session_state('username')
        self.initialize_in_session_state('cwd')
        self.host = host
        self.host_name = host_name if host_name is not None else host
        super().__init__(host=host, *args, **kwargs)

    def run(self, *args, **kwargs):
        if not self.get_from_session_state(f'{self.name}-init'):
            with st.form(key='login_form'):
                st.markdown(f'Login form for {self.host_name} ({self.host})')
                col1, col2 = st.columns(2)
                with col1:
                    username = st.text_input('Username')
                    cwd = st.text_input('Working directory')
                    submitted = st.form_submit_button('Log in')
                with col2:
                    self.password = st.text_input('Password', type='password')
                if submitted:
                    self.login_submit_func(username=username, cwd=cwd, password=self.password)
                    st.experimental_rerun()
        else:
            pass
            # Just run the component logic

    @st.experimental_singleton
    def login_submit_func(self, username, cwd):
        # Log in to system and be sure it works
        with st.spinner('Logging in...'):
            self.save_in_session_state(f'{self.name}-init', True)
            self.save_in_session_state('is_login', True)
            self.save_in_session_state(f'host', self.host)
            self.save_in_session_state(f'host_name', self.host_name)
            self.save_in_session_state(f'username', username)
            # self.save_in_session_state(f'password', self.password)
            self.save_in_session_state(f'cwd', cwd)



            st.success(f'You are logged in to {self.host}')





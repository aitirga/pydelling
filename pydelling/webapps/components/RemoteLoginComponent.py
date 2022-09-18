from pydelling.webapps.components import BaseComponent
import streamlit as st
import time
import paramiko
import extra_streamlit_components as stx


@st.cache(allow_output_mutation=True)
def get_manager():
    return stx.CookieManager()


class RemoteLoginComponent(BaseComponent):
    def __int__(self,
                host: str,
                host_name: str = None,
                username: str = None,
                password: str = None,
                *args,
                **kwargs
                ):
        self.initialize_in_session_state('host')
        self.initialize_in_session_state('host_name')
        self.initialize_in_session_state('username')
        self.initialize_in_session_state('cwd')
        self.initialize_in_session_state('password')
        self.host = host
        self.host_name = host_name if host_name is not None else host
        self.client: paramiko.SSHClient
        self.username = username
        self.password = password
        self.cookie_manager = self.get_manager()

    def run(self, *args, **kwargs):
        if not self.get_from_session_state(f'{self.name}-init'):
            with st.form(key='login_form'):
                st.markdown(f'Login form for {self.host_name} ({self.host})')
                col1, col2 = st.columns(2)
                with col1:
                    self.username = st.text_input('Username')
                    cwd = st.text_input('Working directory')
                    submitted = st.form_submit_button('Log in')
                with col2:
                    self.password = st.text_input('Password', type='password')
                if submitted:
                    self.login_submit_func(username=self.username, cwd=cwd, password=self.password)
                    st.experimental_rerun()
        else:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            # Just run the component logic

    # @st.experimental_memo
    def login_submit_func(self, username, cwd, password):
        # Log in to system and be sure it works
        # self = _self
        with st.spinner('Logging in...'):
            self.cookie_manager: stx.CookieManager = self.get_manager()
            self.save_in_session_state(f'host', self.host)
            self.save_in_session_state(f'host_name', self.host_name)
            self.save_in_session_state(f'username', username)
            self.save_in_session_state(f'password', self.password)
            self.save_in_session_state(f'cwd', cwd)

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(self.host, username=username, password=password)
            st.success(f'You are logged in to {self.host}')
            self.save_in_session_state('is_login', True)
            self.save_in_session_state(f'{self.name}-init', True)
            auth_dict = {
                'host': self.host,
                'host_name': self.host_name,
                'username': username,
                'password': password,
                'cwd': cwd,
            }
            self.cookie_manager.set(f'{self.host_name}-auth', auth_dict)

    def check_connection(self):
        # Check if the connection is still alive
        self.client.connect(self.host,
                            username=self.username,
                            password=self.password,
                            )
        return True






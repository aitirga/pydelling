from pydelling.webapps import WebAppRunner
from pydelling.webapps.components import RemoteLoginComponent
import streamlit as st
import time

class PflotranManagerWebapp(WebAppRunner):
    """This webapp can be used to run and manage PFLOTRAN runs both in local and remote machines."""
    remote_hosts = {
        'tranchete': '192.168.69.34',
        'jureca': '192.168.2.1',
        'transformer': '192.168.2.1',
    }
    def initialize(self):
        self.initialize_in_session_state('is_login', False)
        self.initialize_in_session_state('host', None)
        self.initialize_in_session_state('host_name', None)
        self.initialize_in_session_state('username', None)
        self.initialize_in_session_state('cwd', None)
        self.initialize_in_session_state('cur_status', None)
        st.sidebar.title('PFLOTRAN Simulation Manager')
        self.sidebar_info = st.sidebar.empty()
        if not self.get_from_session_state('is_login'):
            self.sidebar_info.markdown('Please login to continue')
        else:
            with st.sidebar:
                self.sidebar_info.markdown(f' You are logged in to {self.get_from_session_state("host_name")}')
                st.markdown(f'**Username:** {self.get_from_session_state("username")}')
                st.text_input('Working directory',
                              key='cwd-text',
                              value=self.get_from_session_state('cwd'),
                              on_change=self.change_working_directory,
                              )
                st.button('Logout', on_click=self.logout)


    def construct(self):
        # First check if the user is logged in
        self.handle_events()
        if not self.get_from_session_state('is_login'):
            self.login()
        else:
            self.main()

    def login(self):
        self.empty_selectbox = st.empty()
        host_name = self.empty_selectbox.selectbox('Select a login method', ['Tranchete', 'JURECA', 'Transformer'],
                                       on_change=self.show_remote_login,
                                       )
        self.remote_login = RemoteLoginComponent(
            host=self.remote_hosts[host_name.lower()],
            host_name=host_name
        )
        self.show_remote_login()

    def logout(self):
        st.success('You have been logged out, please login again')
        self.set_to_session_state('is_login', False)


    def main(self):
        st.title('PFLOTRAN Simulation Manager')
        st.markdown('This webapp can be used to run and manage PFLOTRAN runs in remote machines.')
        st.markdown(f'You are currently logged in to {self.get_from_session_state("host_name")} using the username {self.get_from_session_state("username")}')

    def show_remote_login(self):
        st.session_state[f'{self.remote_login.name}-init'] = False
        # self.set_to_session_state('is_login', True)

    def event_handler(self, event):
        pass

    def handle_events(self):
        pass

    def change_working_directory(self):
        self.save_in_session_state('cwd', self.get_from_session_state('cwd-text'))


if __name__ == '__main__':
    webapp = PflotranManagerWebapp()
    webapp.run()
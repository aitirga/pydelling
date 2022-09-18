from pydelling.webapps import WebAppRunner
from pydelling.webapps.components import RemoteLoginComponent
import streamlit as st
import time

class PflotranManagerWebapp(WebAppRunner):
    """This webapp can be used to run and manage PFLOTRAN runs both in local and remote machines."""
    remote_ips = {
        'tranchete': '192.168.1.1'
    }
    def initialize(self):
        self.initialize_in_session_state('is_login', False)
        self.initialize_in_session_state('host', None)
        self.initialize_in_session_state('username', None)
        self.initialize_in_session_state('cwd', None)
        st.sidebar.title('PFLOTRAN Simulation Manager')
        self.sidebar_info = st.sidebar.empty()
        if not self.get_from_session_state('is_login'):
            self.sidebar_info.markdown('Please login to continue')
        else:
            self.sidebar_info.markdown(f' You are logged in to {self.get_from_session_state("host")}')
            st.sidebar.markdown(f'**Username:** {self.get_from_session_state("username")}')
            st.sidebar.markdown(f'**Working directory:** {self.get_from_session_state("cwd")}')
            st.sidebar.button('Logout', on_click=self.logout)

    def construct(self):
        # First check if the user is logged in
        self.handle_events()
        if not self.get_from_session_state('is_login'):
            self.login()
        else:
            self.main()

    def login(self):
        self.empty_selectbox = st.empty()
        self.remote_login = RemoteLoginComponent(
            host='test',
        )
        self.empty_selectbox.selectbox('Select a login method', ['Tranchete', 'JURECA', 'Transformer'],
                                       on_change=self.show_remote_login,
                                       )

    def logout(self):
        st.success('You have been logged out, please login again')
        self.set_to_session_state('is_login', False)


    def main(self):
        pass

    def show_remote_login(self):
        st.session_state[f'{self.remote_login.name}-init'] = False
        # self.set_to_session_state('is_login', True)

    def event_handler(self, event):
        pass

    def handle_events(self):
        pass





if __name__ == '__main__':
    webapp = PflotranManagerWebapp()
    webapp.run()
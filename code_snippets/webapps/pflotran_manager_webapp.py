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
        self.initialize_in_session_state('password', None)
        self.initialize_in_session_state('cur_cookie', False)
        self.initialize_in_session_state('cookie_login', False)
        self.initialize_in_session_state('hard_logout', False)
        st.sidebar.title('PFLOTRAN Simulation Manager')
        self.sidebar_info = st.sidebar.empty()

        get_cookies = self.get_manager().get_all()
        for cookie in get_cookies:
            if 'auth' in cookie:
                self.set_to_session_state('host_name', get_cookies[cookie]['host_name'])
                self.set_to_session_state('username', get_cookies[cookie]['username'])
                self.set_to_session_state('cwd', get_cookies[cookie]['cwd'])
                self.set_to_session_state('host', get_cookies[cookie]['host'])
                self.set_to_session_state('password', get_cookies[cookie]['password'])
                self.set_to_session_state('cur_cookie', cookie)
                self.set_to_session_state('is_login', True)
                break

        if self.get_from_session_state('hard_logout'):
            self.set_to_session_state('is_login', False)
            self.set_to_session_state('hard_logout', False)
        if not self.get_from_session_state('is_login'):
            self.sidebar_info.markdown('Please login to continue')
        else:
            with st.sidebar:
                # Check if there is an authenticated cookie

                self.remote = RemoteLoginComponent(
                    host=self.get_from_session_state('host'),
                    host_name=self.get_from_session_state('host_name'),
                    username=self.get_from_session_state('username'),
                    password=self.get_from_session_state('password'),
                    login_node=False,
                )
                # self.remote.check_connection()
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
        if self.get_from_session_state('hard_logout'):
            self.set_to_session_state('is_login', False)
            self.set_to_session_state('hard_logout', False)

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
            host_name=host_name,
            login_node=True,
        )

    def logout(self):
        st.success('You have been logged out, please login again')
        # Delete the cookie
        self.get_manager().delete(self.get_from_session_state('cur_cookie'))
        self.save_in_session_state('hard_logout', True)
        self.save_in_session_state('is_login', False)

    def main(self):
        st.title('PFLOTRAN Simulation Manager')
        st.markdown('This webapp can be used to run and manage PFLOTRAN runs in remote machines.')
        st.markdown(f'You are currently logged in to {self.get_from_session_state("host_name")} using the username {self.get_from_session_state("username")}')

    def show_remote_login(self):
        try:
            st.session_state[f'{self.remote_login.name}-init'] = False
        except:
            pass

    def event_handler(self, event):
        pass

    def handle_events(self):
        pass

    def change_working_directory(self):
        self.save_in_session_state('cwd', self.get_from_session_state('cwd-text'))

    def check_login(self):
        self.remote.check_connection()




if __name__ == '__main__':
    webapp = PflotranManagerWebapp()
    webapp.run()
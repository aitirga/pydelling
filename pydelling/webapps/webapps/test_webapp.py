import streamlit as st

from WebAppRunner import WebAppRunner


class TestWebApp(WebAppRunner):
    def construct(self):
        st.header("Test WebApp")


if __name__ == '__main__':
    test_webapp = TestWebApp()
    test_webapp.run()

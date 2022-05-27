from WebAppRunner import WebAppRunner
import streamlit as st


class TestWebApp(WebAppRunner):
    def construct(self):
        st.header("Test WebApp")


if __name__ == '__main__':
    test_webapp = TestWebApp()
    test_webapp.run()

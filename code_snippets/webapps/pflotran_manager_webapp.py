from pydelling.webapps import WebAppRunner
import streamlit as st


class PflotranManagerWebapp(WebAppRunner):
    def construct(self):
        pass


if __name__ == '__main__':
    webapp = PflotranManagerWebapp()
    webapp.run()
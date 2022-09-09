from pydelling.webapps import WebAppRunner
import streamlit as st

class PflotranRunnerWebapp(WebAppRunner):
    """This webapp allows to run PFLOTRAN simulations on JURECA and visualize the evolution"""
    def construct(self):
        txt = st.text_area('Text to analyze', '''
             It was the best of times, it was the worst of times, it was
             the age of wisdom, it was the age of foolishness, it was
             the epoch of belief, it was the epoch of incredulity, it
             was the season of Light, it was the season of Darkness, it
             was the spring of hope, it was the winter of despair, (...)
             ''', height=1000, disabled=True)


if __name__ == '__main__':
    webapp = PflotranRunnerWebapp()
    webapp.run()

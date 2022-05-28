import pandas as pd

from pydelling.webapps.components import BaseComponent
from pydelling.webapps import WebAppRunner
from pydelling.config import config
import streamlit as st
from pydelling.readers import VTKMeshReader
from pathlib import Path

class InputComponent(BaseComponent):
    """Creates a streamlit block in which the user can drag and drop a file and the resulting object is returned"""
    def __init__(self, input_type: str, key: str, webapp: WebAppRunner=None):
        self.key = key
        if not f'{self.key}_done' in st.session_state:
            st.session_state[f'{self.key}_done'] = False

        self.input_type = input_type
        super().__init__(webapp)


    def run(self):
        """Runs the component"""
        if self.input_type == 'csv':
            self.run_csv_input()
        elif self.input_type == 'vtk':
            self.run_vtk_input()

    def run_csv_input(self):
        with st.form(f"{self.key}_form"):
            st.write("Drop your input data (the data that contains the information to interpolate) in the box below")
            data = st.file_uploader("Upload your data", type=config.globals.webapp.supported_input_data)
            st.markdown('**Advanced upload options:**')
            cols = st.columns([1, 6, 6])
            with cols[0]:
                sep = st.text_input('Separator', ',')
            whitespace_delimiter = st.checkbox('Whitespace delimiter', value=False)
            # Every form must have a submit button.
            submitted = st.form_submit_button("Upload")
            if submitted:
                if whitespace_delimiter:
                    read_data = pd.read_csv(data, delim_whitespace=whitespace_delimiter)
                else:
                    read_data = pd.read_csv(data, sep=sep)
                st.dataframe(read_data.head())
                self._value = read_data
                st.session_state[f'{self.key}_done'] = True
                if not self.key in st.session_state:
                    st.session_state[f'{self.key}'] = read_data


    def run_vtk_input(self):
        with st.form(f'{self.key}_form'):
            st.write("Drop your mesh/centroid data (the mesh/location you want to interpolate into) in the box below")
            st.markdown('This data must be in VTK format')
            data = st.file_uploader("Upload your data", type='vtk')
            new_submit = st.form_submit_button("Upload")
            if new_submit:
                # Write file to temp directory
                with open('temp.vtk', 'wb') as f:
                    f.write(data.getbuffer())
                vtk_mesh = VTKMeshReader('temp.vtk', st_file=True)
                Path('temp.vtk').unlink()
                st.session_state[f'{self.key}'] = vtk_mesh
                st.session_state[f'{self.key}_done'] = True





import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from pydelling.config import config
from pydelling.interpolation import SparseDataInterpolator
from pydelling.webapps import WebAppRunner
from pydelling.webapps.components import InputComponent


class SparseInterpolatorWebapp(WebAppRunner):
    def construct(self):
        if 'input_selection_done' not in st.session_state:
            st.session_state['input_selection_done'] = False
        if 'mesh_input_done' not in st.session_state:
            st.session_state['mesh_input_done'] = False
        if 'interpolation_done' not in st.session_state:
            st.session_state['interpolation_done'] = False

        st.header('Sparse data interpolator')
        st.markdown("""
        This webapp allows you to upload sparse data (A cloud of points) in `|x|y|z|value|` format and interpolate it 
        into a mesh or another target cloud of points.

        These are the steps:

        """)
        st.subheader('Step 1: Upload sparse data')
        InputComponent(input_type='csv', key='input_data')
        if st.session_state['input_data_done']:
            self.input_data: pd.DataFrame = st.session_state['input_data']

            with st.form('select input variables'):
                st.markdown("""Select the columns corresponding to the`x`, `y` and `z` variables and the `target variable`""")
                cols = st.columns(4)
                x_index = None
                y_index = None
                z_index = None
                for col_idx, column_name in enumerate(self.input_data.columns):
                    if 'x' == column_name.lower():
                        x_index = col_idx
                    elif 'y' == column_name.lower():
                        y_index = col_idx
                    elif 'z' == column_name.lower():
                        z_index = col_idx

                with cols[0]:
                    x_col = st.selectbox('x centroid', self.input_data.columns, index=x_index if x_index is not None else 0)
                with cols[1]:
                    y_col = st.selectbox('y centroid', self.input_data.columns, index=y_index if y_index is not None else 0)
                with cols[2]:
                    z_col = st.selectbox('z centroid', self.input_data.columns, index=z_index if z_index is not None else 0)
                with cols[3]:
                    target_col = st.selectbox('target variable', self.input_data.columns, index=len(self.input_data.columns) - 1)
                submit = st.form_submit_button('Select variables')
                if submit:
                    st.success(f'centroid variables: [{x_col}, {y_col}, {z_col}] and target variable: {target_col} selected')
                    st.session_state['input_selection_done'] = True

        if st.session_state['input_selection_done']:
            st.subheader('Step 2: Upload mesh data')
            selected_type = st.selectbox('Select target mesh file format', config.globals.webapps.supported_mesh_data)
            if selected_type == 'csv':
                InputComponent(input_type='csv', key='mesh_input')
                with st.form('select input variables mesh'):
                    st.markdown(
                        """Select the columns corresponding to the`x`, `y` and `z` variables of the centroids""")
                    cols = st.columns(3)
                    x_index = None
                    y_index = None
                    z_index = None
                    for col_idx, column_name in enumerate(self.input_data.columns):
                        if 'x' == column_name.lower():
                            x_index = col_idx
                        elif 'y' == column_name.lower():
                            y_index = col_idx
                        elif 'z' == column_name.lower():
                            z_index = col_idx

                    with cols[0]:
                        x_col = st.selectbox('x centroid', self.input_data.columns,
                                             index=x_index if x_index is not None else 0)
                    with cols[1]:
                        y_col = st.selectbox('y centroid', self.input_data.columns,
                                             index=y_index if y_index is not None else 0)
                    with cols[2]:
                        z_col = st.selectbox('z centroid', self.input_data.columns,
                                             index=z_index if z_index is not None else 0)
                    submit = st.form_submit_button('Select variables')
                    if submit:
                        st.success(
                            f'centroid variables: [{x_col}, {y_col}, {z_col}] selected as the target mesh')
                        st.session_state['input_selection_done'] = True

            elif selected_type == 'vtk':
                InputComponent(input_type='vtk', key='mesh_input')

            elif selected_type == 'fem':
                InputComponent(input_type='fem', key='mesh_input')

        if st.session_state['mesh_input_done']:
            st.subheader('Step 3: Interpolate data')
            input_data = self.input_data[[x_col, y_col, z_col, target_col]]
            if selected_type == 'vtk':
                mesh_data = st.session_state['mesh_input']
                mesh_data = mesh_data.centroids
            elif selected_type == 'csv':
                mesh_data = st.session_state['mesh_input'][[x_col, y_col, z_col]].values
            elif selected_type == 'fem':
                mesh_data = st.session_state['mesh_input'].centroids
            with st.form('interpolation_step'):
                st.markdown("""Choose the interpolation method""")
                interpolation_method = st.selectbox('interpolation method', ['nearest', 'linear', 'cubic'], index=0)

                interpolator = SparseDataInterpolator(interpolation_data=input_data, mesh_data=mesh_data)
                submit = st.form_submit_button()
                if submit:
                    with st.spinner('interpolating data'):
                        interpolator.run(method=interpolation_method)
                    st.success(f'interpolation done')
                    interpolated_data = interpolator.interpolated_data
                    interpolated_block = np.array([[mesh_data[i, 0], mesh_data[i, 1], mesh_data[i, 2], interpolated_data[i]] for i in range(len(mesh_data))])
                    interpolated_block = pd.DataFrame(interpolated_block, columns=['x', 'y', 'z', target_col])
                    st.session_state['interpolation_done'] = True
                    st.session_state['interpolated_data'] = interpolated_block
        if st.session_state['interpolation_done']:
            interpolated_block = st.session_state['interpolated_data']

            st.subheader('Step 4: Download interpolated data')
            st.download_button(
                label="Download data as CSV",
                data=self.convert_df(interpolated_block),
                file_name='interpolated_data.csv',
                mime='text/csv',
            )
            visualize = st.checkbox('Visualize interpolated data')
            if visualize:
                import pyvista as pv
                interpolated_block = st.session_state['interpolated_data']
                pl = pv.Plotter()
                point_cloud = pv.PolyData(interpolated_block[['x', 'y', 'z']].values)
                point_cloud['value'] = interpolated_block[target_col].values
                point_size = point_cloud.bounds
                point_size = np.max(point_size) * 0.0075

                pl.add_mesh(point_cloud, point_size=point_size, style='points')
                pl.background_color = 'white'
                pl.add_scalar_bar(
                    'value',
                    interactive=True, vertical=False,
                    title_font_size=35,
                    label_font_size=30,
                    outline=True, fmt='%10.5f',
                )
                pl.export_html('pyvista.html')  # doctest:+SKIP

                HtmlFile = open("pyvista.html", 'r', encoding='utf-8')
                source_code = HtmlFile.read()
                components.html(source_code, height=750, width=1000)

    @staticmethod
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')


if __name__ == '__main__':
    webapp = SparseInterpolatorWebapp()
    webapp.run()
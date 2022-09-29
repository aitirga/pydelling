import numpy as np

from pydelling.readers import RasterFileReader
from stl import mesh
import logging

logger = logging.getLogger(__name__)

class ClosedStlGenerator(object):
    """This class generates a closed STL from two regular raster files"""
    def __init__(self, bottom_surface: RasterFileReader,
                 top_surface: RasterFileReader = None,
                 aperture: RasterFileReader = None,
    ):
        # Generate needed data
        self.bottom_surface = bottom_surface
        if top_surface is not None:
            self.top_surface = top_surface
        if aperture is not None:
            self.aperture = aperture
        else:
            self.aperture = top_surface - bottom_surface
        assert self.aperture is not None, "Aperture is not defined"
        self.stl_file: mesh.Mesh

    def run(self, output_filename: str = 'closed_stl.stl'):
        """This method runs the closed STL generator"""
        logger.info(f'Generating closed STL based on bottom raster file: {self.bottom_surface.filename} and top raster file: {self.top_surface.filename}')
        # First, we need to generate the vertices
        vertices = []
        # Now, we need to generate the faces
        faces = []
        # First, we need to generate the faces for the bottom surface
        logger.info('Generating faces for the bottom surface')
        for row in range(self.bottom_surface.nrows - 1):
            for col in range(self.bottom_surface.ncols - 1):
                # Generate the face
                # Save the face vertices
                vertex_1 = [self.bottom_surface.x_mesh[row, col], self.bottom_surface.y_mesh[row, col], self.bottom_surface.data[row, col]]
                vertex_2 = [self.bottom_surface.x_mesh[row, col + 1], self.bottom_surface.y_mesh[row, col + 1], self.bottom_surface.data[row, col + 1]]
                vertex_3 = [self.bottom_surface.x_mesh[row + 1, col + 1], self.bottom_surface.y_mesh[row + 1, col + 1], self.bottom_surface.data[row + 1, col + 1]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)

                # Save the face vertices
                vertex_1 = [self.bottom_surface.x_mesh[row, col], self.bottom_surface.y_mesh[row, col], self.bottom_surface.data[row, col]]
                vertex_2 = [self.bottom_surface.x_mesh[row + 1, col + 1], self.bottom_surface.y_mesh[row + 1, col + 1], self.bottom_surface.data[row + 1, col + 1]]
                vertex_3 = [self.bottom_surface.x_mesh[row + 1, col], self.bottom_surface.y_mesh[row + 1, col], self.bottom_surface.data[row + 1, col]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)

        # Now, we need to generate the faces for the top surface
        logger.info('Generating faces for the top surface')
        for row in range(self.top_surface.nrows - 1):
            for col in range(self.top_surface.ncols - 1):
                # Generate the face
                # Save the face vertices
                vertex_1 = [self.top_surface.x_mesh[row, col], self.top_surface.y_mesh[row, col], self.top_surface.data[row, col]]
                vertex_2 = [self.top_surface.x_mesh[row, col + 1], self.top_surface.y_mesh[row, col + 1], self.top_surface.data[row, col + 1]]
                vertex_3 = [self.top_surface.x_mesh[row + 1, col + 1], self.top_surface.y_mesh[row + 1, col + 1], self.top_surface.data[row + 1, col + 1]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)

                # Save the face vertices
                vertex_1 = [self.top_surface.x_mesh[row, col], self.top_surface.y_mesh[row, col], self.top_surface.data[row, col]]
                vertex_2 = [self.top_surface.x_mesh[row + 1, col + 1], self.top_surface.y_mesh[row + 1, col + 1], self.top_surface.data[row + 1, col + 1]]
                vertex_3 = [self.top_surface.x_mesh[row + 1, col], self.top_surface.y_mesh[row + 1, col], self.top_surface.data[row + 1, col]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)

        # Now, we need to generate the four faces that connect the top and bottom surfaces
        # row = 0
        logger.info('Generating faces that connect the top and bottom surfaces')
        for col in range(self.top_surface.ncols - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[0, col], self.top_surface.y_mesh[0, col], self.top_surface.data[0, col]]
            vertex_2 = [self.top_surface.x_mesh[0, col + 1], self.top_surface.y_mesh[0, col + 1], self.top_surface.data[0, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[0, col + 1], self.bottom_surface.y_mesh[0, col + 1], self.bottom_surface.data[0, col + 1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[0, col], self.top_surface.y_mesh[0, col], self.top_surface.data[0, col]]
            vertex_2 = [self.bottom_surface.x_mesh[0, col + 1], self.bottom_surface.y_mesh[0, col + 1], self.bottom_surface.data[0, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[0, col], self.bottom_surface.y_mesh[0, col], self.bottom_surface.data[0, col]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

        # row = nrows - 1
        for col in range(self.top_surface.ncols - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[-1, col], self.top_surface.y_mesh[-1, col], self.top_surface.data[-1, col]]
            vertex_2 = [self.top_surface.x_mesh[-1, col + 1], self.top_surface.y_mesh[-1, col + 1], self.top_surface.data[-1, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[-1, col + 1], self.bottom_surface.y_mesh[-1, col + 1], self.bottom_surface.data[-1, col + 1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[-1, col], self.top_surface.y_mesh[-1, col], self.top_surface.data[-1, col]]
            vertex_2 = [self.bottom_surface.x_mesh[-1, col + 1], self.bottom_surface.y_mesh[-1, col + 1], self.bottom_surface.data[-1, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[-1, col], self.bottom_surface.y_mesh[-1, col], self.bottom_surface.data[-1, col]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

        # col = 0
        for row in range(self.top_surface.nrows - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, 0], self.top_surface.y_mesh[row, 0], self.top_surface.data[row, 0]]
            vertex_2 = [self.top_surface.x_mesh[row + 1, 0], self.top_surface.y_mesh[row + 1, 0], self.top_surface.data[row + 1, 0]]
            vertex_3 = [self.bottom_surface.x_mesh[row + 1, 0], self.bottom_surface.y_mesh[row + 1, 0], self.bottom_surface.data[row + 1, 0]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, 0], self.top_surface.y_mesh[row, 0], self.top_surface.data[row, 0]]
            vertex_2 = [self.bottom_surface.x_mesh[row + 1, 0], self.bottom_surface.y_mesh[row + 1, 0], self.bottom_surface.data[row + 1, 0]]
            vertex_3 = [self.bottom_surface.x_mesh[row, 0], self.bottom_surface.y_mesh[row, 0], self.bottom_surface.data[row, 0]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

        # col = ncols - 1
        for row in range(self.top_surface.nrows - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, -1], self.top_surface.y_mesh[row, -1], self.top_surface.data[row, -1]]
            vertex_2 = [self.top_surface.x_mesh[row + 1, -1], self.top_surface.y_mesh[row + 1, -1], self.top_surface.data[row + 1, -1]]
            vertex_3 = [self.bottom_surface.x_mesh[row + 1, -1], self.bottom_surface.y_mesh[row + 1, -1], self.bottom_surface.data[row + 1, -1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, -1], self.top_surface.y_mesh[row, -1], self.top_surface.data[row, -1]]
            vertex_2 = [self.bottom_surface.x_mesh[row + 1, -1], self.bottom_surface.y_mesh[row + 1, -1], self.bottom_surface.data[row + 1, -1]]
            vertex_3 = [self.bottom_surface.x_mesh[row, -1], self.bottom_surface.y_mesh[row, -1], self.bottom_surface.data[row, -1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)

        vertices = np.array(vertices)

        # Now we need to generate the mesh
        self.stl_file = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(vertices):
            for j in range(3):
                self.stl_file.vectors[i][j] = f[j]

        # Write the mesh to file
        logger.info(f'Writing mesh to {output_filename}')
        self.stl_file.save(output_filename)

    def plot_aperture(self):
        """This method plots the aperture"""
        self.aperture.plot(colorbar_label='Aperture')


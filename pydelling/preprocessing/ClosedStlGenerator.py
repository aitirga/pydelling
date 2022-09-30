import numpy as np

from pydelling.readers import RasterFileReader
from stl import mesh
import logging
from pydelling.utils import create_results_folder

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

    def run(self, output_filename: str = 'closed_stl.stl',
            export_faces=True,
            ):
        """This method runs the closed STL generator"""
        logger.info(f'Generating closed STL based on bottom raster file: {self.bottom_surface.filename} and top raster file: {self.top_surface.filename}')
        # First, we need to generate the vertices
        vertices = []
        # Now, we need to generate the faces
        faces = []
        # First, we need to generate the faces for the bottom surface
        logger.info('Generating faces for the bottom surface')
        faces_dict = {
            'bottom': [],
            'top': [],
            'west': [],
            'east': [],
            'north': [],
            'south': [],
        }
        for row in range(self.bottom_surface.nrows - 1):
            for col in range(self.bottom_surface.ncols - 1):
                # Generate the face
                # Save the face vertices
                vertex_1 = [self.bottom_surface.x_mesh[row, col], self.bottom_surface.y_mesh[row, col], self.bottom_surface.data[row, col]]
                vertex_2 = [self.bottom_surface.x_mesh[row, col + 1], self.bottom_surface.y_mesh[row, col + 1], self.bottom_surface.data[row, col + 1]]
                vertex_3 = [self.bottom_surface.x_mesh[row + 1, col + 1], self.bottom_surface.y_mesh[row + 1, col + 1], self.bottom_surface.data[row + 1, col + 1]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)
                faces_dict['bottom'].append(face_vertices)

                # Save the face vertices
                vertex_1 = [self.bottom_surface.x_mesh[row, col], self.bottom_surface.y_mesh[row, col], self.bottom_surface.data[row, col]]
                vertex_2 = [self.bottom_surface.x_mesh[row + 1, col + 1], self.bottom_surface.y_mesh[row + 1, col + 1], self.bottom_surface.data[row + 1, col + 1]]
                vertex_3 = [self.bottom_surface.x_mesh[row + 1, col], self.bottom_surface.y_mesh[row + 1, col], self.bottom_surface.data[row + 1, col]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)
                faces_dict['bottom'].append(face_vertices)


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
                faces_dict['top'].append(face_vertices)

                # Save the face vertices
                vertex_1 = [self.top_surface.x_mesh[row, col], self.top_surface.y_mesh[row, col], self.top_surface.data[row, col]]
                vertex_2 = [self.top_surface.x_mesh[row + 1, col + 1], self.top_surface.y_mesh[row + 1, col + 1], self.top_surface.data[row + 1, col + 1]]
                vertex_3 = [self.top_surface.x_mesh[row + 1, col], self.top_surface.y_mesh[row + 1, col], self.top_surface.data[row + 1, col]]
                face_vertices = [vertex_1, vertex_2, vertex_3]
                vertices.append(face_vertices)
                faces_dict['top'].append(face_vertices)

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
            faces_dict['north'].append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[0, col], self.top_surface.y_mesh[0, col], self.top_surface.data[0, col]]
            vertex_2 = [self.bottom_surface.x_mesh[0, col + 1], self.bottom_surface.y_mesh[0, col + 1], self.bottom_surface.data[0, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[0, col], self.bottom_surface.y_mesh[0, col], self.bottom_surface.data[0, col]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['north'].append(face_vertices)

        # row = nrows - 1
        for col in range(self.top_surface.ncols - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[-1, col], self.top_surface.y_mesh[-1, col], self.top_surface.data[-1, col]]
            vertex_2 = [self.top_surface.x_mesh[-1, col + 1], self.top_surface.y_mesh[-1, col + 1], self.top_surface.data[-1, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[-1, col + 1], self.bottom_surface.y_mesh[-1, col + 1], self.bottom_surface.data[-1, col + 1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['south'].append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[-1, col], self.top_surface.y_mesh[-1, col], self.top_surface.data[-1, col]]
            vertex_2 = [self.bottom_surface.x_mesh[-1, col + 1], self.bottom_surface.y_mesh[-1, col + 1], self.bottom_surface.data[-1, col + 1]]
            vertex_3 = [self.bottom_surface.x_mesh[-1, col], self.bottom_surface.y_mesh[-1, col], self.bottom_surface.data[-1, col]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['south'].append(face_vertices)

        # col = 0
        for row in range(self.top_surface.nrows - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, 0], self.top_surface.y_mesh[row, 0], self.top_surface.data[row, 0]]
            vertex_2 = [self.top_surface.x_mesh[row + 1, 0], self.top_surface.y_mesh[row + 1, 0], self.top_surface.data[row + 1, 0]]
            vertex_3 = [self.bottom_surface.x_mesh[row + 1, 0], self.bottom_surface.y_mesh[row + 1, 0], self.bottom_surface.data[row + 1, 0]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['west'].append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, 0], self.top_surface.y_mesh[row, 0], self.top_surface.data[row, 0]]
            vertex_2 = [self.bottom_surface.x_mesh[row + 1, 0], self.bottom_surface.y_mesh[row + 1, 0], self.bottom_surface.data[row + 1, 0]]
            vertex_3 = [self.bottom_surface.x_mesh[row, 0], self.bottom_surface.y_mesh[row, 0], self.bottom_surface.data[row, 0]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['west'].append(face_vertices)

        # col = ncols - 1
        for row in range(self.top_surface.nrows - 1):
            # Generate the face
            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, -1], self.top_surface.y_mesh[row, -1], self.top_surface.data[row, -1]]
            vertex_2 = [self.top_surface.x_mesh[row + 1, -1], self.top_surface.y_mesh[row + 1, -1], self.top_surface.data[row + 1, -1]]
            vertex_3 = [self.bottom_surface.x_mesh[row + 1, -1], self.bottom_surface.y_mesh[row + 1, -1], self.bottom_surface.data[row + 1, -1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['east'].append(face_vertices)

            # Save the face vertices
            vertex_1 = [self.top_surface.x_mesh[row, -1], self.top_surface.y_mesh[row, -1], self.top_surface.data[row, -1]]
            vertex_2 = [self.bottom_surface.x_mesh[row + 1, -1], self.bottom_surface.y_mesh[row + 1, -1], self.bottom_surface.data[row + 1, -1]]
            vertex_3 = [self.bottom_surface.x_mesh[row, -1], self.bottom_surface.y_mesh[row, -1], self.bottom_surface.data[row, -1]]
            face_vertices = [vertex_1, vertex_2, vertex_3]
            vertices.append(face_vertices)
            faces_dict['east'].append(face_vertices)

        vertices = np.array(vertices)

        # Now we need to generate the mesh
        self.stl_file = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(vertices):
            for j in range(3):
                self.stl_file.vectors[i][j] = f[j]

        # Create STL meshes for the faces
        self.faces_stl = {}
        for face in faces_dict:
            vertices = np.array(faces_dict[face])
            self.faces_stl[face] = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(vertices):
                for j in range(3):
                    self.faces_stl[face].vectors[i][j] = f[j]

        self.export_stl(output_filename=output_filename, faces=export_faces)



    def export_stl(self, output_filename, faces=False):
        """
        Exports the generated closed STL
        """
        from pathlib import Path
        results_folder = create_results_folder()
        output_filename = Path(output_filename)
        logger.info(f'Writing mesh to {output_filename}')
        self.stl_file.save(str(results_folder / output_filename))
        if faces:
            for face in self.faces_stl:
                face_filename = Path(f"{output_filename.stem}-{face}.stl")
                logger.info(f'Writing {face} face STL mesh to outputs folder')
                self.faces_stl[face].save(str(results_folder / face_filename))





    def plot_aperture(self):
        """This method plots the aperture"""
        self.aperture.plot(colorbar_label='Aperture')


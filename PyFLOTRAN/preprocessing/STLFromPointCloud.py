import vtkmodules.vtkCommonDataModel

from PyFLOTRAN.preprocessing import BasePreprocessing
from PyFLOTRAN.config import config
from PyFLOTRAN.utils.decorators import set_run
import open3d as o3d
import pandas as pd
import numpy as np
import logging
import vtk

logger = logging.getLogger(__file__)

class STLFromPointCloud(BasePreprocessing):
    preprocessed_data: np.ndarray
    is_run: bool = False
    stl_mesh: o3d.geometry.TriangleMesh

    def __init__(self, data: pd.DataFrame = None, filename: str = None):
        super().__init__(data, filename)
        logger.info("Preprocessing cloud of points")
        assert self.is_data_ok(), "Data is not properly set-up"
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.data)
        self.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5,
                                                          max_nn=75))

    @set_run
    def run(self, method="ball_pivoting", *args, **kwargs):
        if config.stl_from_point_cloud.stl_strategy:
            method = config.stl_from_point_cloud.stl_strategy
        if method == 'ball_pivoting' or method == 'ball pivoting':
            if 'ball_radius' in kwargs:
                self.run_ball_pivoting(ball_radius=kwargs['ball_radius'], **kwargs)
            else:
                self.run_ball_pivoting()

        if method == 'poisson':
            self.run_poisson_reconstruction()

    def is_data_ok(self):
        if isinstance(self.data, pd.DataFrame):
            self.preprocessed_data = self.data.values
            return True
        elif isinstance(self.data, np.ndarray):
            self.preprocessed_data = self.data
            return True
        else:
            return False

    def visualize_point_cloud(self):
        o3d.visualization.draw_geometries([self.point_cloud])

    def visualize_stl_mesh(self):
        assert self.is_run, "Run the self.run() method before visualizing the results"
        o3d.visualization.draw_geometries([self.stl_mesh])

    @set_run
    def run_ball_pivoting(self, ball_radius=None, simplify_mesh=False, **kwargs):
        logger.info("Running the ball pivoting algorithm to recreate the stl surface")
        if not ball_radius and not config.stl_from_point_cloud.ball_radius:
            distances = self.point_cloud.compute_nearest_neighbor_distance()
            self.avg_distance = np.mean(distances)
            self.ball_radius = 3 * self.avg_distance
            self.stl_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd=self.point_cloud,
                radii=o3d.utility.DoubleVector([self.ball_radius, self.ball_radius * 2]),
                **config.stl_from_point_cloud.arguments,
            )
        else:
            self.ball_radius = ball_radius if ball_radius else config.stl_from_point_cloud.ball_radius
            self.stl_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd=self.point_cloud,
                radii=o3d.utility.DoubleVector([self.ball_radius, self.ball_radius * 2]),
                **config.stl_from_point_cloud.arguments,
            )
        if simplify_mesh or config.stl_from_point_cloud.simplify_mesh:
            self.simplify_mesh()

    def make_mesh_watertight(self,
                             hole_size=1,
                             filename="reconstructed_mesh.ply",
                             write=True):
        """This method reads a ply mesh from an external file and converts it into a watertight mesh
        """

        reader = vtk.vtkPLYReader()
        reader.SetFileName(filename)
        reader.Update()
        polydata = reader.GetOutput()
        fill = vtk.vtkFillHolesFilter()
        fill.SetInputData(polydata)
        fill.SetHoleSize(hole_size)
        fill.Update()
        self.vtk_filled = fill.GetOutput()
        if write:
            writer = vtk.vtkPLYWriter()
            writer.SetInputData(self.vtk_filled)
            writer.SetFileName(f"{filename}-watertight.ply")
            writer.Write()
        return fill.GetOutput()

    def reduce_mesh(self, target_value=100000):
        self.stl_mesh = self.stl_mesh.reduce_mesh(target_value)

    def simplify_mesh(self, target_value=100000):
        self.stl_mesh.remove_degenerate_triangles()
        self.stl_mesh.remove_duplicated_triangles()
        self.stl_mesh.remove_duplicated_vertices()
        self.stl_mesh.remove_non_manifold_edges()

    @set_run
    def run_poisson_reconstruction(self, depth=8, **kwargs):
        logger.info("Running the poisson reconstruction method to recreate the stl surface")
        self.point_cloud.orient_normals_towards_camera_location(self.point_cloud.get_center())
        self.point_cloud.normals = o3d.utility.Vector3dVector(- np.asarray(self.point_cloud.normals))
        self.stl_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud,
                                                                                  depth=depth,
                                                                                  **kwargs if kwargs else {},
                                                                                  **config.stl_from_point_cloud.arguments,
                                                                                  )[0]
        # bbox = self.point_cloud.get_axis_aligned_bounding_box()
        # self.stl_mesh = self.stl_mesh.crop(bbox)

    def to_ply(self, filename="reconstructed_mesh.ply"):
        """Writes the mesh in ply format
        """
        assert self.is_run, "a mesh reconstruction technique needs to be run first"
        o3d.io.write_triangle_mesh(filename, self.stl_mesh)
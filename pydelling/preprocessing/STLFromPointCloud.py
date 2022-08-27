# import logging
# from pathlib import Path
#
# import numpy as np
# import open3d as o3d
# import pandas as pd
# import vtk
#
# from pydelling.config import config
# from pydelling.preprocessing import BasePreprocessing
# from pydelling.utils.decorators import set_run
#
# logger = logging.getLogger(__name__)
#
#
# class STLFromPointCloud(BasePreprocessing):
#     preprocessed_data: np.ndarray
#     is_run: bool = False
#     stl_mesh: o3d.geometry.TriangleMesh
#
#     def __init__(self, data: pd.DataFrame = None, filename: str = None):
#         super().__init__(data, filename)
#
#
#     @set_run
#     def run(self, method="ball_pivoting", *args, **kwargs):
#         logger.info("Preprocessing cloud of points")
#         assert self.is_data_ok(), "Data is not properly set-up"
#         self.point_cloud = o3d.geometry.PointCloud()
#         self.point_cloud.points = o3d.utility.Vector3dVector(self.data)
#         self.point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.5,
#                                                           max_nn=75))
#         if config.stl_from_point_cloud.stl_strategy:
#             method = config.stl_from_point_cloud.stl_strategy
#         if method == 'ball_pivoting' or method == 'ball pivoting':
#             if 'ball_radius' in kwargs:
#                 self.run_ball_pivoting(ball_radius=kwargs['ball_radius'], **kwargs)
#             else:
#                 self.run_ball_pivoting()
#
#         if method == 'poisson':
#             self.run_poisson_reconstruction()
#
#     def is_data_ok(self):
#         if isinstance(self.data, pd.DataFrame):
#             self.preprocessed_data = self.data.values
#             return True
#         elif isinstance(self.data, np.ndarray):
#             self.preprocessed_data = self.data
#             return True
#         else:
#             return False
#
#     def visualize_point_cloud(self):
#         o3d.visualization.draw_geometries([self.point_cloud])
#
#     def visualize_stl_mesh(self, stl_mesh=None):
#         if stl_mesh is not None:
#             o3d.visualization.draw_geometries([stl_mesh])
#         else:
#             assert self.is_run, "Run the self.run() method before visualizing the results"
#             o3d.visualization.draw_geometries([self.stl_mesh])
#
#     @set_run
#     def run_ball_pivoting(self, ball_radius=None, simplify_mesh=False, **kwargs):
#         logger.info("Running the ball pivoting algorithm to recreate the stl surface")
#         if not ball_radius and not config.stl_from_point_cloud.ball_radius:
#             distances = self.point_cloud.compute_nearest_neighbor_distance()
#             self.avg_distance = np.mean(distances)
#             self.ball_radius = 3 * self.avg_distance * config.stl_from_point_cloud.buff if config.stl_from_point_cloud.buff else 1.0
#             self.stl_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#                 pcd=self.point_cloud,
#                 radii=o3d.utility.DoubleVector([self.ball_radius, self.ball_radius * 2]),
#                 **config.stl_from_point_cloud.arguments,
#             )
#         else:
#             self.ball_radius = ball_radius if ball_radius else config.stl_from_point_cloud.ball_radius
#             self.stl_mesh: o3d.geometry.TriangleMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#                 pcd=self.point_cloud,
#                 radii=o3d.utility.DoubleVector([self.ball_radius, self.ball_radius * 2]),
#                 **config.stl_from_point_cloud.arguments,
#             )
#         if simplify_mesh or config.stl_from_point_cloud.simplify_mesh:
#             self.simplify_mesh()
#         return self.stl_mesh
#
#
#     def make_mesh_watertight(self,
#                              hole_size=None,
#                              filename=None,
#                              write=True):
#         """This method reads a ply mesh from an external file and converts it into a watertight mesh
#         """
#         self.to_ply(filename="temp.ply")
#         logger.info(f"Making the mesh read from {filename} watertight")
#         reader = vtk.vtkPLYReader()
#         if filename is None:
#             filename = Path.cwd() / "output/temp.ply"
#         reader.SetFileName(str(filename))
#         reader.Update()
#         polydata = reader.GetOutput()
#         fill = vtk.vtkFillHolesFilter()
#         fill.SetInputData(polydata)
#         hole_size = hole_size if hole_size else config.stl_from_point_cloud.watertight_mesh.hole_size if config.stl_from_point_cloud.watertight_mesh.hole_size else 1.0
#         fill.SetHoleSize(hole_size)
#         fill.Update()
#         self.vtk_filled = fill.GetOutput()
#         if write:
#             writer = vtk.vtkPLYWriter()
#             writer.SetInputData(self.vtk_filled)
#             writer.SetFileName(f"temp-watertight.ply")
#             writer.Write()
#         self.stl_mesh = o3d.io.read_triangle_mesh("temp-watertight.ply")
#         print(self.stl_mesh)
#         # Path.unlink(Path("temp-watertight.ply"))
#         # Path.unlink(filename)
#
#         return fill.GetOutput()
#
#     def reduce_mesh(self, target_value=100000):
#         self.stl_mesh = self.stl_mesh.reduce_mesh(target_value)
#         return self.stl_mesh
#
#
#     def simplify_mesh(self, target_value=100000):
#         self.stl_mesh.remove_degenerate_triangles()
#         self.stl_mesh.remove_duplicated_triangles()
#         self.stl_mesh.remove_duplicated_vertices()
#         self.stl_mesh.remove_non_manifold_edges()
#         return self.stl_mesh
#
#
#     @set_run
#     def run_poisson_reconstruction(self, depth=8, **kwargs):
#         logger.info("Running the poisson reconstruction method to recreate the stl surface")
#         self.point_cloud.orient_normals_towards_camera_location(self.point_cloud.get_center())
#         self.point_cloud.normals = o3d.utility.Vector3dVector(- np.asarray(self.point_cloud.normals))
#         self.stl_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.point_cloud,
#                                                                                   depth=depth,
#                                                                                   **kwargs if kwargs else {},
#                                                                                   **config.stl_from_point_cloud.arguments,
#                                                                                   )[0]
#         return self.stl_mesh
#
#     def to_ply(self, filename="reconstructed_mesh.ply"):
#         """Writes the mesh in ply format
#         """
#         # assert self.is_run, "a mesh reconstruction technique needs to be run first"
#         o3d.io.write_triangle_mesh(str(self.output_directory / filename), self.stl_mesh)
#         logger.info(f"Writing to {str(self.output_directory / filename)}")
#
#     def to_stl(self, filename="reconstructed_mesh.stl"):
#         """Writes the mesh in stl format
#         """
#         # assert self.is_run, "a mesh reconstruction technique needs to be run first"
#         self.stl_mesh = self.stl_mesh.compute_vertex_normals()
#         logger.info(f"Writing to {str(self.output_directory / filename)}")
#         o3d.io.write_triangle_mesh(str(self.output_directory / filename), self.stl_mesh)
#
#     def read_stl(self, filename):
#         """Reads a stl file and converts it into a o3d mesh"""
#         self.stl_mesh = o3d.io.read_triangle_mesh(filename)
#
#     def crop_bounding_box(self, bbox=None):
#         """
#         Uses original cloud of point bounding box to crop the STL mesh
#         """
#         if bbox is None:
#             logger.info("Computing bounding box and cropping the mesh")
#             bbox = self.point_cloud.get_axis_aligned_bounding_box()
#         else:
#             logger.info(f"Computing bounding box at {bbox} and cropping the mesh")
#             bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=bbox[0], max_bound=bbox[1])
#
#         self.stl_mesh = self.stl_mesh.crop(bbox)
#         return self.stl_mesh
#
#     @staticmethod
#     def create_output_directory():
#         """Creates an ./output folder on the working directory
#         """
#         Path(Path.cwd() / "output").mkdir(parents=True, exist_ok=True)
#
#     @property
#     def output_directory(self):
#         self.create_output_directory()
#         return Path.cwd() / "output"
#

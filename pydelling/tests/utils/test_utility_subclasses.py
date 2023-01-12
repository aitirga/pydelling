import unittest
from pydelling.utils.utility_subclasses import SemistructuredFinder
import numpy as np

class GeometryUtilsCase(unittest.TestCase):
    def test_semistructured_finder_2d(self):
        n_clusters_x = 10
        n_clusters_y = 10
        n_points = 25
        cluster_centers = [[x, y] for x in range(n_clusters_x) for y in range(n_clusters_y)]
        points = [[x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02)] for x, y in cluster_centers for _ in range(n_points)]
        points = np.array(points)

        cluster_finder = SemistructuredFinder(points, eps=0.05, min_samples=5)
        self.assertEqual(cluster_finder.n_clusters, n_clusters_x * n_clusters_y)

    def test_semistructured_finder_3d(self):
        n_clusters_x = 10
        n_clusters_y = 10
        n_points = 25
        cluster_centers = [[x, y, z] for x in range(n_clusters_x) for y in range(n_clusters_y) for z in range(10)]
        points = [[x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02), z] for x, y, z in cluster_centers for _ in
                  range(n_points)]
        points = np.array(points)

        cluster_finder = SemistructuredFinder(points, eps=0.05, min_samples=20, projection_axis='z')
        self.assertEqual(cluster_finder.n_clusters, n_clusters_x * n_clusters_y)

        # # Plot the clusters
        # import matplotlib.pyplot as plt
        # for cluster in cluster_finder.clusters:
        #     # Transform ((x1, y1), (x2, y2), ...) to (x1, x2, ...), (y1, y2, ...)
        #     x, y, z = zip(*cluster)
        #     plt.scatter(x, y)
        # plt.show()

        # # Plot just the first cluster
        # first_cluster = cluster_finder.clusters[5]
        # x, y, z = zip(*first_cluster)
        # import matplotlib.pyplot as plt
        # plt.scatter(x, y)
        # plt.show()


    def test_semistructured_finder_point_columns(self):
        n_clusters_x = 10
        n_clusters_y = 10
        n_points = 25
        cluster_centers = [[x, y, z] for x in range(n_clusters_x) for y in range(n_clusters_y) for z in range(10)]
        points = [[x + np.random.normal(0, 0.02), y + np.random.normal(0, 0.02), z] for x, y, z in cluster_centers for _ in
                  range(n_points)]
        points = np.array(points)

        cluster_finder = SemistructuredFinder(points, eps=0.05, min_samples=20, projection_axis='z')
        target_point = np.array([5.0, 5, 5])
        closest_cluster = cluster_finder.get_closest_cluster_from_point(target_point)
        self.assertLess(np.linalg.norm(target_point - np.array(closest_cluster).mean()), 0.3)
        # Plot the clusters with high alpha, the selected cluster opaque and the point
        # import matplotlib.pyplot as plt
        # x, y, z = zip(*closest_cluster)
        # plt.scatter(x, y)
        # plt.scatter(target_point[0], target_point[1], alpha=1)
        # plt.show()

        closest_cluster = cluster_finder.get_closest_cluster_from_xy(x=5.0, y=5.0)
        self.assertLess(np.linalg.norm(target_point - np.array(closest_cluster).mean()), 0.3)

        # Get the point ids instead of the points
        closest_cluster = cluster_finder.get_closest_point_ids_from_point(target_point)







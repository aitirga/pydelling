from pydelling.preprocessing.mesh_preprocessor import MeshPreprocessor


class UnstructuredMeshReader(MeshPreprocessor):
    def __init__(self, nodes, vertices):
        super().__init__()
        self.nodes = nodes
        self.vertices = vertices

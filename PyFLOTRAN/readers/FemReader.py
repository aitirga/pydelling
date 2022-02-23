from ..preprocessing.MeshPreprocessor import MeshPreprocessor


class FemReader(MeshPreprocessor):
    def __init__(self, filename, **kwargs):
        super().__init__()

    def open_file(self, filename, **kwargs):
        pass

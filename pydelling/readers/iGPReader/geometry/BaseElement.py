from pydelling.config import config


class BaseElement:
    def __init__(self, node_ids, node_coords, element_type_n, local_id, centroid_coords=None):
        self.nodes = node_ids  # Node id set
        self.coords = node_coords  # Coordinates of each node
        self.centroid_coords = centroid_coords
        self.type = 'BaseElement'
        self.n_type = element_type_n  # Number of node_ids for an element
        self.local_id = local_id  # Element id
        self.faces = {}  # Dictionary to store face information

    def __repr__(self):
        # print("### Element info ###")
        # print(f"Element ID: {self.local_id}")
        # print(f"Number of nodes: {self.n_type}")
        # print(f"Element type: {self.type}")
        # print(f"Node list: {self.nodes}")
        # print("### End element info ###")
        #
        # print("### Face info ###")
        # for face in self.faces:
        #     print(f"{face}: {self.faces[face].coords}")
        # print("### End face info ###")
        return f"{self.type} {self.local_id}"


    def print_element_info(self):
        print("### Element info ###")
        print(f"Element ID: {self.local_id}")
        print(f"Number of nodes: {self.n_type}")
        print(f"Element type: {self.type}")
        print(f"Node list: {self.nodes}")
        print("### End element info ###")

    def print_face_info(self):
        print("### Face info ###")
        for face in self.faces:
            print(f"{face}: {self.faces[face].coords}")
        print("### End face info ###")

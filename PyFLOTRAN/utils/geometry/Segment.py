from . import Line, Point, BasePrimitive


class Segment(Line):
    def __init__(self, p1, p2):
        super().__init__(p1, p2)
        self.p1 = Point(p1)
        self.p2 = Point(p2)
        self.displacement = self.p2 - self.p1

    def __repr__(self):
        return f"Segment(p1: {self.p1}, p2: {self.p2})"

    def __str__(self):
        return f"Segment(p1: {self.p1}, p2: {self.p2})"

    @property
    def length(self):
        return self.p1.distance(self.p2)


    def intersect(self, primitive: BasePrimitive):
        from .intersections import intersect_plane_segment

        if primitive.__class__.__name__ == "Plane":
            return intersect_plane_segment(segment=self, plane=primitive)

        else:
            raise NotImplementedError(f"Intersection of {type(self)} with {type(primitive)} is not implemented")

    def contains(self, point: Point):
        return self.p1.distance(point) + self.p2.distance(point) == self.length


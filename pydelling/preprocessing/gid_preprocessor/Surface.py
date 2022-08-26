from ._AbstractGidObject import _AbstractGidObject
from .Line import Line
from .Point import Point
import logging
from typing import List
import logging
from typing import List

from .Line import Line
from .Point import Point
from ._AbstractGidObject import _AbstractGidObject

logger = logging.getLogger(__name__)


class Surface(_AbstractGidObject):
    local_id: int = 1
    global_id: int = 1
    surfaces: List = []

    def __init__(self, lines: List[Line]):
        self.local_id = Surface.local_id
        self.lines = lines
        self.id = None
        Surface.local_id += 1

    def add(self):
        self.id = Surface.global_id
        logger.info(f'Adding surface {self.id} connecting {self.lines}')
        Surface.global_id += 1
        Surface.surfaces.append(self)
        export_str = 'Mescape Geometry Create NurbsSurface\n'
        for line in self.lines:
            export_str += f'{line.id} '
        export_str += '\n'
        return export_str

    def extrude(self, start_point: Point, end_point: Point, end_object):
        logger.info(f'Extruding {self} using direction {start_point.coords} -> {end_point.coords}')
        export_str = f'Mescape Utilities Copy Surfaces DoExtrude {end_object} MaintainLayers Translation FNoJoin {start_point.coords_comma} FNoJoin {end_point.coords_comma}\n'
        export_str += f'{self.id}\n'
        extra_surfaces = 5
        extra_lines = 2 * len(self.lines)
        extra_points = len(self.lines)
        Surface.global_id += extra_surfaces
        Line.global_id += extra_lines
        Point.global_id += extra_points
        return export_str

    def __repr__(self):
        if self.id:
            return f"Surface {self.id}"
        else:
            return f"Surface (local) {self.local_id}"


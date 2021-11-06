from __future__ import annotations
from ._AbstractGidObject import _AbstractGidObject
from .Point import Point
from .Line import Line
import logging
import functools
logger = logging.getLogger(__file__)
from typing import Union, List

class GidObject(object):
    """
    This class could be inherited to create new GiD objects
    It stores different GiD objects and generates batch files to generate the geometry automatically on GiD
    """
    def __init__(self):
        self.batch_commands = ''
        self.pipeline = []
        logger.info('A new GidObject has been created')

    def construct(self, *args, **kwargs):
        pass

    def add(self, objects: Union[List[_AbstractGidObject], _AbstractGidObject, GidObject]):
        """
        Adds an object of type AbstractGidObject to the current GidObject instance depending on its type
        """
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            assert isinstance(obj, _AbstractGidObject) or isinstance(obj, GidObject), 'The given object must be a Point, Line, Surface, Volume or another GidObject'
            if isinstance(obj, Point):
                self.pipeline.append({'add': obj})

            if isinstance(obj, Line):
                self.pipeline.append({'add': obj})

            if isinstance(obj, GidObject):
                self.import_gidobject(obj)

    def join(self, point_1: Point, point_2: Point):
        join_line: Line = Line(point_1, point_2)
        self.add(join_line)

    def import_gidobject(self, gidobject):
        gidobject.construct()
        self.pipeline += gidobject.pipeline

    def run(self, filename='gid_batch.bch', add_escape=False, internal=False):
        """
        Processes the construct method and generates the bash file
        """
        logger.info('Processing GidObject by runnning the "construct()" method')
        self.construct()  # Generates the pipeline
        for step in self.pipeline:
            keys = list(step.keys())
            step_type = keys[0]
            if step.get('internal'):
                method = getattr(self, step_type)
            else:
                method = getattr(step[step_type], step_type)
            if add_escape:
                method = self.add_escape(method)
            self.batch_commands += method(*step.get('args', {}), **step.get('kwargs', {}))

        self.batch_commands += 'escape'
        if internal:
            return self.batch_commands
        else:
            with open(filename, 'w') as write_file:
                write_file.write(self.batch_commands)


    @staticmethod
    def add_escape(func):
        @functools.wraps(func)
        def decorator(*args, **kwargs):
            func_return = func(*args, **kwargs)
            func_return += 'escape\n'
            return func_return
        return decorator


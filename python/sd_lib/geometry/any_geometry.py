# coding: utf-8

from sd_lib.geometry.geometry import Geometry
from sd_lib.geometry.constants import ANY_SHAPE

class AnyGeometry(Geometry):
    '''
    This is a class for creating and using AnyGeometry for Labels.
    '''
    @staticmethod
    def geometry_name():
        return ANY_SHAPE


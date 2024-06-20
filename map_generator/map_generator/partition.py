import copy
from typing import Union
from itertools import product
from abc import ABC, abstractmethod
import numpy as np


class AbstractPoint(ABC):

    @abstractmethod
    def get_coord(self) -> tuple:
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class Point(AbstractPoint):
    """pointer to coordinates on the board"""

    def __init__(self, row, col):
        self.__row = row
        self.__col = col

    def get_coord(self):
        return self.__row, self.__col

    def __add__(self, other):
        row1, col1 = self.get_coord()
        row2, col2 = other.get_coord()
        return Point(row1 + row2, col1 + col2)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, constant):
        row, col = self.get_coord()
        return Point(row * constant, col * constant)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return -1 * other + self

    def __eq__(self, other):
        return self.get_coord() == other.get_coord()

    def __hash__(self):
        return hash(self.get_coord())

    def __str__(self):
        return str(self.get_coord())

    def __repr__(self):
        return self.__str__()


class AbstractPartition(ABC):
    num_row = 10
    num_col = 9
    @abstractmethod
    def __init__(self):
        self.__points: set[Point] = set()

    @staticmethod
    def get_horizontal_line(row:int):
        return set([Point(row, i) for i in range(AbstractPartition.num_col)])

    @staticmethod
    def get_vertical_line(col:int):
        return set([Point(i, col) for i in range(AbstractPartition.num_row)])

    @property
    def points(self):
        return self.__points

    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.points)

class PartitionTriangleVertical(AbstractPartition):
    def __init__(self, direction: str = "up"):
        super().__init__()
        self.__direction = direction
        self.__points: set[Point] = self.__get_shape_points()

    def __get_shape_points(self):
        _points = set()
        row_idx = self.num_row
        col_start, col_end = 1, self.num_col - 1
        while(col_start == col_end):
            _points |= [Point(row_idx, _y) for _y in range(col_start, col_end)]
            col_start += 1
            col_end -= 1
        return _points

    def __str__(self):
        return "PartitionTriangleVertical"
    
    def __repr__(self):
        return self.__str__()


class PartitionBigX(AbstractPartition):
    def __init__(self):
        self.__points: set[Point] = self.__get_shape_points()

    def __get_shape_points(self):
        pass

    def __str__(self):
        return "PartitionBigX"


class PartitionSquare(AbstractPartition):
    def __init__(self, start_point: Point, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height
        self.start_point = start_point
        self.__points: list[Point] = self.__get_shape_points()

    def __get_shape_points(self):
        return [
            self.start_point + Point(i, j)
            for i, j in product(range(self.width), range(self.height))
        ]

    def __str__(self):
        return f"PartitionSquare at {self.start_point} with width {self.width} and height {self.height}"


class PartitionCenterPlus(AbstractPartition):
    def __init__(self):
        super().__init__()
        self.__points: list[Point] = self.__get_shape_points()

    def __get_shape_points(self):
        return set([
            *self.get_horizontal_line(4, 10),
            *self.get_horizontal_line(5, 10),
            *self.get_vertical_line(4, 9),
        ])

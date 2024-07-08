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

    @property
    def points(self):
        return set()

    @abstractmethod
    def __init__(self):
        pass

    @staticmethod
    def get_horizontal_line(row: int):
        return set([Point(row, i) for i in range(AbstractPartition.num_col)])

    @staticmethod
    def get_vertical_line(col: int):
        return set([Point(i, col) for i in range(AbstractPartition.num_row)])

    def to_numpy_index(self):
        _points = self.points
        return np.array([i.get_coord() for i in _points]).T.tolist()

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __eq__(self):
        pass

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.points)


class PartitionTriangle(AbstractPartition):
    def __init__(self, direction: str = "up"):
        super().__init__()
        self.__direction = direction
        self.__points: set[Point] = self.__get_shape_points()

    @property
    def points(self):
        return self.__points

    def __get_shape_points(self):
        if self.__direction == "up":
            return self.__get_shape_upward_triangle()
        elif self.__direction == "down":
            return self.__get_shape_downward_triangle()
        elif self.__direction == "left":
            return self.__get_shape_leftward_triangle()
        elif self.__direction == "right":
            return self.__get_shape_rightward_triangle()
        else:
            raise ValueError(f"Do not support direction = {self.__direction}")

    def __get_shape_upward_triangle(self):
        _points = set()
        row_idx = self.num_row - 1
        col_start, col_end = 1, self.num_col - 1
        while col_start <= col_end:
            _points |= set([Point(row_idx, _y) for _y in range(col_start, col_end)])
            col_start += 1
            col_end -= 1
            row_idx -= 1
        return _points

    def __get_shape_downward_triangle(self):
        _points = set()
        row_idx = 0
        col_start, col_end = 1, self.num_col - 1
        while col_start <= col_end:
            _points |= set([Point(row_idx, _y) for _y in range(col_start, col_end)])
            col_start += 1
            col_end -= 1
            row_idx += 1
        return _points

    def __get_shape_leftward_triangle(self):
        _points = set()
        col_idx = 0
        row_start, row_end = 1, self.num_row - 1
        while row_start <= row_end:
            _points |= set([Point(_x, col_idx) for _x in range(row_start, row_end)])
            row_start += 1
            row_end -= 1
            col_idx += 1
        return _points

    def __get_shape_rightward_triangle(self):
        _points = set()
        col_idx = self.num_col - 1
        row_start, row_end = 1, self.num_row - 1
        while row_start <= row_end:
            _points |= set([Point(_x, col_idx) for _x in range(row_start, row_end)])
            row_start += 1
            row_end -= 1
            col_idx -= 1
        return _points

    def __eq__(self, another):
        return self.__direction == another.__direction

    def __str__(self):
        return "PartitionTriangle"

    def __repr__(self):
        return self.__str__()


class PartitionBigX(AbstractPartition):
    def __init__(self):
        self.__points: set[Point] = self.__get_shape_points()

    @property
    def points(self):
        return self.__points

    def __get_shape_points(self):
        _points = []
        for i in range(self.num_row // 2):
            _points.append(Point(i, i))
            _points.append(Point(self.num_row - (i + 1), self.num_col - (i + 1)))
            _points.append(Point(i, self.num_col - (i + 1)))
            _points.append(Point(self.num_row - (i + 1), i))

        return set(_points)

    def __eq__(self, another):
        return isinstance(another) == PartitionBigX

    def __str__(self):
        return "PartitionBigX"


class PartitionSquare(AbstractPartition):
    def __init__(self, start_point: Point, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height
        self.start_point = start_point
        self.__points: set[Point] = self.__get_shape_points()

    @property
    def points(self):
        return self.__points

    def __get_shape_points(self):
        return set(
            [
                self.start_point + Point(i, j)
                for i, j in product(range(self.height), range(self.width))
            ]
        )

    def __eq__(self, another):
        return (
            self.start_point == another.start_point
            and self.width == another.width
            and self.height == another.height
        )

    def __str__(self):
        return f"PartitionSquare at {self.start_point} with width {self.width} and height {self.height}"


class PartitionCenterPlus(AbstractPartition):
    def __init__(self):
        super().__init__()
        self.__points: list[Point] = self.__get_shape_points()

    @property
    def points(self):
        return self.__points

    def __get_shape_points(self):
        return set(
            [
                *self.get_horizontal_line(4),
                *self.get_horizontal_line(5),
                *self.get_vertical_line(4),
            ]
        )
    
    def __eq__(self, another):
        return (isinstance(another) == PartitionCenterPlus)

    def __str__(self):
        return "PartitionCenterPlus"


class AbstractScissor(ABC):
    def __init__(self) -> None:
        pass

    @staticmethod
    def divide(div_type, *args):
        if div_type == "square5x5":
            return [
                PartitionSquare(Point(0, 0), 5, 5),
                PartitionSquare(Point(5, 0), 4, 5),
                PartitionSquare(Point(0, 5), 4, 5),
                PartitionSquare(Point(5, 4), 5, 5),
            ]
        elif div_type == "bigX":
            return [
                PartitionBigX(),
                PartitionTriangle("up"),
                PartitionTriangle("down"),
                PartitionTriangle("left"),
                PartitionTriangle("right"),
            ]
        elif div_type == "centerPlus":
            return [
                PartitionCenterPlus(),
                PartitionSquare(Point(0, 0), 4, 4),
                PartitionSquare(Point(6, 0), 4, 4),
                PartitionSquare(Point(0, 5), 4, 4),
                PartitionSquare(Point(6, 5), 4, 4),
            ]
        elif div_type == "square4x4":
            return [
                PartitionSquare(Point(0, 0), 4, 6),
                PartitionSquare(Point(6, 0), 4, 4),
                PartitionSquare(Point(0, 4), 5, 6),
                PartitionSquare(Point(6, 4), 5, 4),
            ]
        else:
            raise ValueError(f"Do not support div_type = {div_type}")


class Scissor(AbstractScissor):
    pass

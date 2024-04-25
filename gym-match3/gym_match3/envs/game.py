import copy
from typing import Union
from itertools import product
from functools import wraps
from abc import ABC, abstractmethod
import numpy as np

from gym_match3.envs.constants import GameObject, mask_immov_mask, need_to_match


class OutOfBoardError(IndexError):
    pass


class ImmovableShapeError(ValueError):
    pass


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
    """ pointer to coordinates on the board"""

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


class Cell(Point):
    def __init__(self, shape, row, col):
        self.__shape = shape
        super().__init__(row, col)

    @property
    def shape(self):
        return self.__shape

    @property
    def point(self):
        return Point(*self.get_coord())

    def __eq__(self, other):
        eq_shape = self.shape == other.shape
        eq_points = super().__eq__(other)
        return eq_shape and eq_points

    def __hash__(self):
        return hash((self.shape, self.get_coord()))


class AbstractBoard(ABC):

    @property
    @abstractmethod
    def board(self):
        pass

    @property
    @abstractmethod
    def board_size(self):
        pass

    @property
    @abstractmethod
    def n_shapes(self):
        pass

    @abstractmethod
    def swap(self, point1: Point, point2: Point):
        pass

    @abstractmethod
    def set_board(self, board: np.ndarray):
        pass

    @abstractmethod
    def move(self, point: Point, direction: Point):
        pass

    @abstractmethod
    def shuffle(self, random_state=None):
        pass

    @abstractmethod
    def get_shape(self, point: Point):
        pass

    @abstractmethod
    def delete(self, points):
        pass

    @abstractmethod
    def get_line(self, ind):
        pass

    @abstractmethod
    def put_line(self, ind, line):
        pass

    @abstractmethod
    def put_mask(self, mask, shapes):
        pass


def check_availability_dec(func):
    @wraps(func)
    def wrapped(self, *args):
        self._check_availability(*args)
        return func(self, *args)

    return wrapped


class Board(AbstractBoard):
    """ board for match3 game"""

    def __init__(self, rows, columns, n_shapes, immovable_shape=-1):
        self.__rows = rows
        self.__columns = columns
        self.__n_shapes = n_shapes
        self.__immovable_shape = immovable_shape
        self.__board = None  # np.ndarray

        if 0 <= immovable_shape < n_shapes:
            raise ValueError('Immovable shape has to be less or greater than n_shapes')

    def __getitem__(self, indx: Point):
        self.__check_board()
        self.__validate_points(indx)
        if isinstance(indx, Point):
            return self.board.__getitem__(indx.get_coord())
        else:
            raise ValueError('Only Point class supported for getting shapes')

    def __setitem__(self, value, indx: Point):
        self.__check_board()
        # print(indx)
        self.__validate_points(indx)
        if isinstance(indx, Point):
            self.__board.itemset(indx.get_coord(), value)
        else:
            raise ValueError('Only Point class supported for setting shapes')

    def __str__(self):
        if isinstance(self.board, np.ndarray):
            return str(self.board)
        return self.board.board

    @property
    def immovable_shape(self):
        return self.__immovable_shape

    @property
    def board(self):
        self.__check_board()
        return self.__board

    @property
    def board_size(self):
        if self.__is_board_exist():
            rows, cols = self.board.shape
        else:
            rows, cols = self.__rows, self.__columns
        return rows, cols

    def set_board(self, board: np.ndarray):
        self.__validate_board(board)
        self.__board = board.astype(float)

    def shuffle(self, random_state=None):
        moveable_mask = self.board != self.immovable_shape
        board_ravel = self.board[moveable_mask]
        np.random.seed(random_state)
        np.random.shuffle(board_ravel)
        self.put_mask(moveable_mask, board_ravel)

    def __check_board(self):
        if not self.__is_board_exist():
            raise ValueError('Board is not created')

    @property
    def n_shapes(self):
        return self.__n_shapes

    @check_availability_dec
    def swap(self, point1: Point, point2: Point):
        point1_shape = self.get_shape(point1)
        point2_shape = self.get_shape(point2)
        self.put_shape(point2, point1_shape)
        self.put_shape(point1, point2_shape)

    def put_shape(self, shape, point: Point):
        self[point] = shape

    def move(self, point: Point, direction: Point):
        self._check_availability(point)
        new_point = point + direction
        self.swap(point, new_point)

    def __is_board_exist(self):
        existence = (self.__board is not None)
        return existence

    def __validate_board(self, board: np.ndarray):
        # self.__validate_max_shape(board) # No check here because of multi tile
        self.__validate_board_size(board)

    def __validate_board_size(self, board: np.ndarray):
        provided_board_shape = board.shape
        right_board_shape = self.board_size
        correct_shape = (provided_board_shape == right_board_shape)
        if not correct_shape:
            raise ValueError('Incorrect board shape: '
                             f'{provided_board_shape} vs {right_board_shape}')

    def __validate_max_shape(self, board: np.ndarray):
        if np.all(np.isnan(board)):
            return
        provided_max_shape = np.nanmax(board)

        right_max_shape = self.n_shapes
        if provided_max_shape > right_max_shape:
            raise ValueError('Incorrect shapes of the board: '
                             f'{provided_max_shape} vs {right_max_shape}')

    def get_shape(self, point: Point):
        return self[point]

    def __validate_points(self, *args):
        for point in args:
            is_valid = self.__is_valid_point(point)
            if not is_valid:
                raise OutOfBoardError(f'Invalid point: {point.get_coord()}')

    def __is_valid_point(self, point: Point):
        row, col = point.get_coord()
        board_rows, board_cols = self.board_size
        correct_row = ((row + 1) <= board_rows) and (row >= 0)
        correct_col = ((col + 1) <= board_cols) and (col >= 0)
        return correct_row and correct_col

    def _check_availability(self, *args):
        for p in args:
            shape = self.get_shape(p)
            if shape == self.immovable_shape:
                raise ImmovableShapeError

    def delete(self, points: set):
        self._check_availability(*points)
        coordinates = tuple(np.array([i.get_coord() for i in points]).T.tolist())
        self.__board[coordinates] = np.nan
        return self

    def get_line(self, ind, axis=1):
        return np.take(self.board, ind, axis=axis)
    
    def get_monster(self):
        return [Point(i, j) for i, j in product(range(self.board_size[0]), range(self.board_size[1])) if self.get_shape(Point(i, j)) in GameObject.monsters]

    def put_line(self, ind, line: np.ndarray):
        # TODO: create board with putting lines on arbitrary axis
        self.__validate_line(ind, line)
        # self.__validate_max_shape(line)
        self.__board[:, ind] = line
        return self

    def put_mask(self, mask, shapes):
        self.__validate_mask(mask)
        # self.__validate_max_shape(shapes)
        self.__board[mask] = shapes
        return self

    def __validate_mask(self, mask):
        if np.any(self.board[mask] == self.immovable_shape):
            raise ImmovableShapeError

    def __validate_line(self, ind, line):
        immove_mask = mask_immov_mask(self.board[:, ind], self.immovable_shape)
        new_immove_mask = mask_immov_mask(np.array(line), self.immovable_shape)
        # print(immove_mask)
        # print(new_immove_mask)
        if not np.array_equal(immove_mask, new_immove_mask):
            raise ImmovableShapeError


class RandomBoard(Board):

    def set_random_board(self, random_state=None):
        board_size = self.board_size

        np.random.seed(random_state)
        board = np.random.randint(
            low=GameObject.color1,
            high=self.n_shapes + 1,
            size=board_size)
        self.set_board(board)
        return self


class CustomBoard(Board):

    def __init__(self, board: np.ndarray, n_shapes: int):
        columns, rows = board.shape
        super().__init__(columns, rows, n_shapes)
        self.set_board(board)


class AbstractSearcher(ABC):
    def __init__(self, board_ndim):
        self.__directions = self.__get_directions(board_ndim)
        self.__disco_directions = self.__get_disco_directions(board_ndim)
        self.__bomb_directions = self.__get_bomb_directions(board_ndim)
        self.__missile_directions = self.__get_missile_directions(board_ndim)
        self.__plane_directions = self.__get_plane_directions(board_ndim)
        self.__power_up_cls = [GameObject.power_disco] * len(self.__disco_directions) + [GameObject.power_bomb] * len(self.__bomb_directions) + [GameObject.power_missile_h, GameObject.power_missile_v] + [GameObject.power_plane] * len(self.__plane_directions) + [-1] * len(self.__directions)

    @staticmethod
    def __get_directions(board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(2)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = 1
            directions[ind][1][ind] = -1
        return directions
    
    @staticmethod
    def __get_disco_directions(board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(4)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = -2
            directions[ind][1][ind] = -1
            directions[ind][2][ind] = 1
            directions[ind][3][ind] = 2
        return directions

    @staticmethod
    def __get_plane_directions(board_ndim):
        directions = [
            [
                [0, 1],
                [1, 0],
                [1, 1]
            ]
        ]
        return directions

    @staticmethod
    def __get_bomb_directions(board_ndim):
        directions_T = [
            [[0 for _ in range(board_ndim)] for _ in range(4)]
            for _ in range(5)
        ]
        for ind in range(len(directions_T)):
            directions_T[ind][0][0] = -1
            directions_T[ind][1][0] = 1
            directions_T[ind][2][1] = -1
            directions_T[ind][3][1] = 1
        for ind in range(1, 5):
            coeff = int(ind > 2) * 2
            directions_T[ind][0 + coeff][ind < 3] = -1 + (ind % 2) * 2
            directions_T[ind][1 + coeff][ind < 3] = -1 + (ind % 2) * 2

        directions_L = [
            [[0 for _ in range(board_ndim)] for _ in range(4)]
            for _ in range(4)
        ]
        for ind in range(4):
            coeff = ind % 2 * 2
            directions_L[ind][0 + coeff][ind % 2] = -2 if 0 < ind and ind < 3 else 2
            directions_L[ind][1 + coeff][ind % 2] = -1 if 0 < ind and ind < 3 else 1
            
            directions_L[(ind + 1) % 4][0 + coeff][ind % 2] = -2 if 0 < ind and ind < 3 else 2
            directions_L[(ind + 1) % 4][1 + coeff][ind % 2] = -1 if 0 < ind and ind < 3 else 1

        return directions_T + directions_L

    @staticmethod
    def __get_missile_directions(board_ndim):
        directions = [
            [[0 for _ in range(board_ndim)] for _ in range(3)]
            for _ in range(board_ndim)
        ]
        for ind in range(board_ndim):
            directions[ind][0][ind] = -2
            directions[ind][1][ind] = -1
            directions[ind][2][ind] = 1
        return directions
    
    def get_power_up_type(self, ind):
        return self.__power_up_cls[ind]

    @property
    def directions(self):
        return self.__disco_directions + self.__bomb_directions + self.__missile_directions + self.__plane_directions + self.__directions

    @staticmethod
    def points_generator(board: Board):
        rows, cols = board.board_size
        points = [Point(i, j) for i, j in product(range(rows), range(cols))]
        for point in points:
            if board[point] == board.immovable_shape or not need_to_match(board[point]):
                continue
            else:
                yield point

    def axis_directions_gen(self):
        for axis_dirs in self.directions:
            yield axis_dirs

    def directions_gen(self):
        for axis_dirs in self.directions:
            for direction in axis_dirs:
                yield direction


class AbstractMatchesSearcher(ABC):

    @abstractmethod
    def scan_board_for_matches(self, board: Board):
        pass


class MatchesSearcher(AbstractSearcher):

    def __init__(self, length, board_ndim):
        self.__3length, self.__4length, self.__5length = range(2, 5)
        super().__init__(board_ndim)

    def scan_board_for_matches(self, board: Board):
        matches = set()
        new_power_ups = dict()
        for point in self.points_generator(board):
            to_del, to_add = self.__get_match3_for_point(board, point)
            # print(_)
            if to_del:
                matches.update(to_del)
                new_power_ups.update(to_add)

        return matches, new_power_ups

    def __get_match3_for_point(self, board: Board, point: Point):
        shape = board.get_shape(point)
        match3_list = []
        power_up_list: dict[Point, int] = {}
        for neighbours, length, idx in self.__generator_neighbours(board, point):
            filtered = self.__filter_cells_by_shape(shape, neighbours)
            if len(filtered) == length:
                match3_list.extend(filtered)

                if length > 2 and idx != -1 and isinstance(point, Point):
                    if point in power_up_list.keys():
                        power_up_list[point] = max(power_up_list[point], self.get_power_up_type(idx))
                    else:
                        power_up_list[point] = self.get_power_up_type(idx)

        if len(match3_list) > 0:
            match3_list.append(Cell(shape, *point.get_coord()))

        return match3_list, power_up_list

    def __generator_neighbours(self, board: Board, point: Point):
        for idx, axis_dirs in enumerate(self.directions):
            new_points = [point + Point(*dir_) for dir_ in axis_dirs]
            try:
                yield [Cell(board.get_shape(new_p), *new_p.get_coord())
                       for new_p in new_points], len(axis_dirs), idx
            except OutOfBoardError:
                continue
            finally:
                yield [], 0, -1

    @staticmethod
    def __filter_cells_by_shape(shape, *args):
        return list(filter(lambda x: x.shape == shape, *args))


class AbstractPowerUpActivator(ABC):
    @abstractmethod
    def activate_power_up(self, power_up_type: int, point: Point, board: Board):
        pass


class PowerUpActivator(AbstractPowerUpActivator):
    def __init__(self):
        self.__bomb_affect = self.__get_bomb_affect()
        self.__plane_affect = self.__get_plane_affect()
        
    def activate_power_up(self, power_up_type: int, point: Point, directions, board: Board):
        brokens = set()
        point2 = point + directions
        shape1 = board.get_shape()
        shape2 = board.get_shape()

        if shape1 in GameObject.powers and shape2 in GameObject.powers:
            # Merge power_up
            pass
        elif shape1 in GameObject.powers:
            self.__activate_not_merge(shape1, point, board)
        elif shape2 in GameObject.powers:
            self.__activate_not_merge(shape2, point, board)

        return brokens
    
    def __activate_not_merge(self, power_up_type: int, point: Point, board: Board):
        pass

    @staticmethod
    def __get_plane_affect():
        affects = [[0 for _ in range(2)] for _ in range (4)]
        affects[0][0] = 1
        affects[1][0] = -1
        affects[2][1] = 1
        affects[3][1] = -1

        return affects

    @staticmethod
    def __get_bomb_affect():
        affects = [[i - 3, j - 3] for i, j in product(range(5), range(5))]

        return affects


class AbstractMovesSearcher(ABC):

    @abstractmethod
    def search_moves(self, board: Board):
        pass


class MovesSearcher(AbstractMovesSearcher, MatchesSearcher):

    def search_moves(self, board: Board, all_moves=False):
        possible_moves = set()
        for point in self.points_generator(board):
            possible_moves_for_point = self.__search_moves_for_point(
                board, point)
            possible_moves.update(possible_moves_for_point)
            if len(possible_moves_for_point) > 0 and not all_moves:
                break
        return possible_moves

    def __search_moves_for_point(self, board: Board, point: Point):
        # contain tuples of point and direction
        possible_moves = set()
        for direction in self.directions_gen():
            try:
                board.move(point, Point(*direction))
                matches, _ = self.scan_board_for_matches(board)
                # inverse move
                board.move(point, Point(*direction))
            except (OutOfBoardError, ImmovableShapeError):
                continue
            if len(matches) > 0:
                possible_moves.add((point, tuple(direction)))
        return possible_moves


class AbstractFiller(ABC):

    @abstractmethod
    def move_and_fill(self, board):
        pass


class Filler(AbstractFiller):

    def __init__(self, random_state=None):
        self.__random_state = random_state

    def move_and_fill(self, board: Board):
        self.__move_nans(board)
        self.__fill(board)

    def __move_nans(self, board: Board):
        _, cols = board.board_size
        for col_ind in range(cols):
            line = board.get_line(col_ind)
            if np.any(np.isnan(line)):
                new_line = self._move_line(line, board.immovable_shape)
                board.put_line(col_ind, new_line)
            else:
                continue

    @staticmethod
    def _move_line(line, immovable_shape):
        new_line = np.zeros_like(line)
        num_of_nans = np.isnan(line).sum()
        immov_mask = mask_immov_mask(line, immovable_shape)
        nans_mask = np.isnan(line)
        new_line = np.zeros_like(line)
        new_line = np.where(immov_mask, line, new_line)

        num_putted = 0
        for ind, shape in enumerate(new_line):

            if shape != immovable_shape and num_putted < num_of_nans:
                new_line[ind] = np.nan
                num_putted += 1
                if num_putted == num_of_nans:
                    break

        spec_mask = nans_mask | immov_mask
        regular_values = line[~spec_mask]
        new_line[(new_line == 0)] = regular_values
        return new_line

    def __fill(self, board):
        is_nan_mask = np.isnan(board.board)
        num_of_nans = is_nan_mask.sum()

        np.random.seed(self.__random_state)
        new_shapes = np.random.randint(
            low=GameObject.color1, high=board.n_shapes + 1, size=num_of_nans)
        board.put_mask(is_nan_mask, new_shapes)


class AbstractPowerUpFactory(ABC):
    @abstractmethod
    def get_power_up_type(matches):
        pass

    
class PowerUpFactory(AbstractPowerUpFactory, AbstractSearcher):
    def __init__(self, board_ndim):
        super().__init__(board_ndim)



class AbstractMonster(ABC):
    def __init__(self, relax_interval, setup_interval, hp = 30):
        self.__hp = hp
        self.__progress = 0
        self.__relax_interval = relax_interval
        self.__setup_interval = setup_interval

    @abstractmethod
    def act(self):
        self.__progress += 1

    @abstractmethod
    def attacked(self, damage):
        self.__hp -= damage


class DameMonster(AbstractMonster):
    def __init__(self, relax_interval=8, setup_interval=3, hp=30, dame=3, cancel_dame=5):
        super().__init__(relax_interval, setup_interval, hp)
        self.__damage = dame

        self.__cancel = cancel_dame
        self.__cancel_dame = cancel_dame

    def act(self):
        super().act()
        if self.__cancel <= 0:
            self.__progress = 0
            self.__hp += self.__cancel # __cancel <= 0
            self.__cancel = self.__cancel_dame
            return None
        
        if self.__progress > self.__relax_interval + self.__setup_interval:
            self.__progress = 0
            return {
                "damage": self.__damage
            }
        
        return None

    def attacked(self, damage):
        if self.__setup_interval < self.__progress and \
            self.__progress <= self.__relax_interval:
            self.__cancel -= damage
        else:
            super().attacked(damage)


class BoxMonster(AbstractMonster):
    def __init__(self, box_mons_type, relax_interval=7, hp=30):
        super().__init__(relax_interval, 0, hp)
        self.__box_monster_type = box_mons_type

    def act(self):
        super().act()
        if self.__progress > self.__relax_interval + self.__setup_interval:
            self.__progress = 0
            if self.__box_monster_type == GameObject.monster_box_box:
                return {
                    "box": GameObject.blocker_box
                }
            if self.__box_monster_type == GameObject.monster_box_bomb:
                return {
                    "box": GameObject.blocker_bomb
                }
            if self.__box_monster_type == GameObject.monster_box_thorny:
                return {
                    "box": GameObject.blocker_thorny
                }
            if self.__box_monster_type == GameObject.monster_box_both:
                return {
                    "box": GameObject.blocker_bomb if np.random.uniform(0, 1.0) <= 0.5 else GameObject.blocker_thorny
                }
            

class BombBlocker(DameMonster):
    def __init__(self, relax_interval=3, setup_interval=0, hp=5, dame=2, cancel_dame=5):
        super().__init__(relax_interval, setup_interval, hp, dame, cancel_dame)


class ThornyBlocker(DameMonster):
    def __init__(self, relax_interval=8, setup_interval=3, hp=30, dame=3, cancel_dame=5):
        super().__init__(relax_interval, setup_interval, hp, dame, cancel_dame)


class AbstractGame(ABC):

    @abstractmethod
    def start(self, board):
        pass

    @abstractmethod
    def swap(self, point, point2):
        pass


class Game(AbstractGame):
    def __init__(self, rows, columns, n_shapes, length,
                 player_hp=1,
                 all_moves=False,
                 immovable_shape=-1,
                 random_state=None):
        self.board = Board(
            rows=rows,
            columns=columns,
            n_shapes=n_shapes)
        self.__player_hp = player_hp
        self.__random_state = random_state
        self.__immovable_shape = immovable_shape
        self.__all_moves = all_moves
        self.__mtch_searcher = MatchesSearcher(length=length, board_ndim=2)
        self.__mv_searcher = MovesSearcher(length=length, board_ndim=2)
        self.__filler = Filler(random_state=random_state)
        self.__pu_activator = PowerUpActivator()

    def play(self, board: Union[np.ndarray, None]):
        self.start(board)
        while True:
            try:
                input_str = input()
                coords = input_str.split(', ')
                a, b, a1, b1 = [int(i) for i in coords]
                self.swap(Point(a, b), Point(a1, b1))
            except KeyboardInterrupt:
                break

    def start(self, board: Union[np.ndarray, None, Board]):
        # TODO: check consistency of movable figures and n_shapes
        if board is None:
            rows, cols = self.board.board_size
            board = RandomBoard(rows, cols, self.board.n_shapes)
            board.set_random_board(random_state=self.__random_state)
            board = board.board
            self.board.set_board(board)
        elif isinstance(board, np.ndarray):
            self.board.set_board(board)
        elif isinstance(board, Board):
            self.board = board
        self.__operate_until_possible_moves()

        return self

    def __start_random(self):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes)
        tmp_board.set_random_board(random_state=self.__random_state)
        super().start(tmp_board.board)

    def swap(self, point: Point, point2: Point):
        direction = point2 - point
        score = self.__move(point, direction)
        return score

    def __move(self, point: Point, direction: Point):
        score = 0

        matches, new_power_ups = self.__check_matches(
            point, direction)
        matches.extend()
        if len(matches) > 0:
            score += len(matches)

            self.board.move(point, direction)
            self.board.delete(matches)
            ###
            for _point, _shape in new_power_ups.items():
                print(_point)
                self.board.put_shape(_point, _shape)
            ###
            self.__filler.move_and_fill(self.board)
            score += self.__operate_until_possible_moves()

        return score

    def __check_matches(self, point: Point, direction: Point):
        tmp_board = self.__get_copy_of_board()
        tmp_board.move(point, direction)
        matches, new_power_ups = self.__mtch_searcher.scan_board_for_matches(tmp_board)
        brokes, 
        return matches, new_power_ups

    def __get_copy_of_board(self):
        return copy.deepcopy(self.board)

    def __operate_until_possible_moves(self):
        """
        scan board, then delete matches, move nans, fill
        repeat until no matches and appear possible moves
        """
        score = self.__scan_del_mvnans_fill_until()
        self.__shuffle_until_possible()
        return score

    def __get_matches(self):
        return self.__mtch_searcher.scan_board_for_matches(self.board)
    
    def __activate_power_up(self, power_up_type: int, point: Point):
        return self.__pu_activator.activate_power_up(power_up_type, point, self.board)

    def __get_possible_moves(self):
        return self.__mv_searcher.search_moves(
            self.board,
            all_moves=self.__all_moves)

    def __scan_del_mvnans_fill_until(self):
        score = 0
        matches, _ = self.__get_matches()
        score += len(matches)
        while len(matches) > 0:
            self.board.delete(matches)
            self.__filler.move_and_fill(self.board)
            matches, _ = self.__get_matches()
            score += len(matches)
        return score

    def __shuffle_until_possible(self):
        possible_moves = self.__get_possible_moves()
        while len(possible_moves) == 0:
            self.board.shuffle(self.__random_state)
            self.__scan_del_mvnans_fill_until()
            possible_moves = self.__get_possible_moves()
        return self


class RandomGame(Game):

    def start(self, random_state=None, *args, **kwargs):
        rows, cols = self.board.board_size
        tmp_board = RandomBoard(rows, cols, self.board.n_shapes)
        tmp_board.set_random_board(random_state=random_state)
        super().start(tmp_board.board)

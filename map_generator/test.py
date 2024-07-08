import numpy as np
import time

from gym_match3.envs.constants import GameObject
from gym_match3.envs.game import Board
from map_generator.utils import crossover

template_board = [
    np.array([
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1],
    ]),
    np.array([
        [-1, -1, -1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, -1, -1, -1],
    ]),
]

start_time = time.time()

parent1, parent2 = Board(10, 9, 5), Board(10, 9, 5)

print("init board", time.time() - start_time)

start_time = time.time()

parent1.set_board(template_board[0])
parent2.set_board(template_board[1])

print("set board", time.time() - start_time)

start_time = time.time()

child1, child2 = crossover(parent1, parent2)

print("crossover", time.time() - start_time)
print(child1.board)
print(child2.board)

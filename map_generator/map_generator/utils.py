import random
import copy

import numpy as np

from gym_match3.envs.game import Board
from .partition import Scissor

ways_to_cut = ["square5x5", "bigX", "centerPlus", "square4x4"]


def crossover(parent1: Board, parent2: Board, crossover_probability: float = 0.5):
    """
    crossover two boards
    :param parent1:
    :param parent2:
    :return: new 2 children board
    """
    div_type = random.sample(ways_to_cut, 1)[0]
    pieces = Scissor.divide(div_type=div_type)
    num_keep = int(crossover_probability * len(pieces))

    crossover_gene = [1] * num_keep + [0] * (len(pieces) - num_keep)
    random.shuffle(crossover_gene)

    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

    for i in range(len(crossover_gene)):
        if crossover_gene[i] == 1:
            coordinates = tuple(
                np.array([i.get_coord() for i in pieces[i].points]).T.tolist()
            )
            child1.board[coordinates] = copy.deepcopy(parent2.board[coordinates])
            child2.board[coordinates] = copy.deepcopy(parent1.board[coordinates])

    return child1, child2

def mutate(board: Board):
    """
    mutate board
    :param board:
    :return: new mutated board
    """
    pass


def get_fitness(board: Board):
    """
    get fitness of board
    :param board:
    :return: fitness
    """
    pass


def create_individual():
    """
    create random individuals
    :return:
    """
    pass

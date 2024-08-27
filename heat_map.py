from gym_match3.envs.match3_env import Match3Env
import wandb
import cProfile, pstats
from PPO import *
import cProfile, pstats
import argparse


def make_heatmap(model_id):
    agent = Agent()
    agent.load_heatmap(model_id)
    print(agent.moves_dict) # the keys are the moves, and the values are how many times the move has been played


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()

    make_heatmap(args.model)
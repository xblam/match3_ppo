from gym_match3.envs.match3_env import Match3Env
import wandb
import cProfile, pstats
from PPO import *
import cProfile, pstats
import argparse


def make_heatmap(model_id):
    agent = Agent()
    agent.load_heatmap(model_id)
    print(agent.win_list)


if __name__ == '__main__':
    make_heatmap(124)
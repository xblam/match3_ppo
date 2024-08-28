import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from gym_match3.envs.match3_env import Match3Env
from PPO import *
import argparse

def dict_to_array(action_dict):
    heatmap_arr = np.zeros((10,9))
    for action, value in action_dict.items():
        if action < 80:
            row = action//8
            col = (action%8)
            heatmap_arr[row][col] += value
            heatmap_arr[row][col+1] += value
        elif action >= 80:
            action = action-80
            row = action//9
            col = (action%9)
            heatmap_arr[row][col] += value
            heatmap_arr[row+1][col] += value
    print(heatmap_arr)
    return heatmap_arr

def make_heatmap(model_id):
    agent = Agent()
    agent.load_heatmap(model_id)
    heatmap_arr = dict_to_array(agent.moves_dict)
    plt.figure(figsize=(6,7.5))
    sns.heatmap(heatmap_arr, annot=True, cmap="Blues", cbar=False, fmt=".0f")
    plt.title("heatmap of bot's moves")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str)
    args = parser.parse_args()

    make_heatmap(args.model)
import argparse
import torch

from gym_match3.envs.match3_env import Match3Env
from gym_match3.envs.levels import Match3Levels, LEVELS
from training.ppo import PPO
from training.m3_model.m3_cnn import M3CnnFeatureExtractor

def get_args():
    parser = argparse.ArgumentParser('BEiT fine-tuning and evaluation script for image classification', add_help=False)
    # Rollout Data
    parser.add_argument('--n_steps', type=int, default=32, metavar='n_steps',
                        help='rollout data length (default: 32)')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.003)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)

    return parser.parse_args()

args = get_args()
env = Match3Env(30)

print(env.observation_space)
print(env.action_space)

PPO_trainer = PPO(
    policy="CnnPolicy",
    env=env,
    learning_rate=args.lr,
    n_steps=args.n_steps,
    policy_kwargs={
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "features_extractor_class": M3CnnFeatureExtractor,
        "features_extractor_kwargs": {
            "mid_channels": 8,
            "out_channels": 161,
            "num_first_cnn_layer": 4
        },
        "optimizer_class": torch.optim.Adam,
        "share_features_extractor": False
    },
    device="cuda"
)

while True:
    PPO_trainer.collect_rollouts(PPO_trainer.env, PPO_trainer.rollout_buffer, PPO_trainer.n_steps)
    PPO_trainer.train()
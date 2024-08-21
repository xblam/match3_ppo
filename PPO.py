import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from gym_match3.envs.match3_env import Match3Env
import argparse
import wandb
import cProfile, pstats

DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")
if (T.cuda.is_available()):
    print("RUNNING WITH CUDA---------->")

class PPOMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def get_memory(self):
        return self.states, self.actions, self.probs, self.vals, self.rewards, self.dones 

class ActorNetwork(nn.Module):
    # the asterisk just means that we are unpacking whatever dimensions the input dims is
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.actor_cnn = nn.Sequential(
            nn.Conv2d(in_channels=26, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.actor_fc = nn.Sequential(
            nn.Linear(128*10*9, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 161),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        dist = self.actor_cnn(state)
        dist = dist.view(dist.size(0),-1)
        dist = self.actor_fc(dist)
        dist = Categorical(dist)
        return dist    


class CriticNetwork(nn.Module):
    def __init__(self): 
        super(CriticNetwork, self).__init__()
        self.critic_cnn = nn.Sequential(
            nn.Conv2d(in_channels=26, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.critic_fc = nn.Sequential(
            nn.Linear(128*10*9, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        value = self.critic_cnn(state)
        value = value.view(value.size(0), -1)
        value = self.critic_fc(value)
        return value

class Agent:
    def __init__(self, n_epochs = 10):        
        self.counter_folder = "ppo_state_dicts"
        counter_file = "ppo_state_dicts/ppo_run_counter.txt"
        self.run_id = self.read_counter(f'{counter_file}')
        self.increment_counter(counter_file)
        self.gamma = .99 
        self.policy_clip = .2
        self.n_epochs = n_epochs
        self.gae_lambda = .95

        self.actor = ActorNetwork().to(DEVICE)
        self.critic = CriticNetwork().to(DEVICE)
        self.memory = PPOMemory()
       
    def read_counter(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try: return int(file.read().strip())
                except ValueError: return 0
        else: return 0

    def increment_counter(self, file_path):
        with open(file_path, 'w') as file:
            file.write(str(self.run_id+1))

    def save_model(self):
        checkpoint = {'actor_state': self.actor.state_dict(), 'critic_state': self.critic.state_dict(), 'actor_optimizer': self.actor.optimizer.state_dict(), 'critic_optimizer': self.critic.optimizer.state_dict()}
        os.makedirs(self.counter_folder, exist_ok=True)
        file_path = os.path.join(self.counter_folder, f"{self.run_id}_state_dict.pth")
        T.save(checkpoint, file_path)

    def load_model(self, model_num):
        file_path = os.path.join(self.counter_folder, f"{model_num}_state_dict.pth")
        state_dict = T.load(file_path)
        self.actor.load_state_dict(state_dict['actor_state'])
        self.critic.load_state_dict(state_dict['critic_state'])
        self.actor.optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic.optimizer.load_state_dict(state_dict['critic_optimizer'])

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
        
    def choose_action(self, observation, infos):
        observation = observation.unsqueeze(0).to(DEVICE)
        dist, value = self.actor(observation), self.critic(observation)
        valid_moves = T.tensor(infos['action_space']).to(DEVICE)
        masked_dist = dist.probs*valid_moves 
        if masked_dist.sum() == 0:
            masked_dist = valid_moves
        masked_dist = T.distributions.Categorical(probs=masked_dist)
        action = masked_dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self, end_game_reward): # we can definitely split this stuff into multiple batches and then train on the batches by themselves using multiprocessing
        state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr = self.memory.get_memory()

        reward_arr[-1] += end_game_reward # we will add this to the last move

        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                a_t += discount*(reward_arr[k] + self.gamma*vals_arr[k+1]*\
                        (1-int(dones_arr[k])) - vals_arr[k])
                discount *= self.gamma*self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage).to(DEVICE)

        # use all the lists from the games to update the model
        values = T.tensor(vals_arr).to(DEVICE)
        states = [tensor.to(DEVICE) for tensor in state_arr]
        states = T.stack(states)
        old_probs = T.tensor(old_prob_arr).to(DEVICE)
        actions = T.tensor(action_arr).to(DEVICE)

        dist = self.actor(states)
        critic_value = self.critic(states)

        critic_value = T.squeeze(critic_value)

        new_probs = dist.log_prob(actions)
        # prob_ratio = new_probs.exp() / old_probs.exp()
        prob_ratio = (new_probs - old_probs).exp()
        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage
        actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

        returns = advantage+values
        critic_loss = ((returns-critic_value)**2).mean()

        total_loss = actor_loss + 0.5*critic_loss
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        self.memory.clear_memory()
        return actor_loss, critic_loss
 
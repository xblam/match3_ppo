import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import time

DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")
if (T.cuda.is_available()):
    print("RUNNING WITH CUDA---------->")

class PPOMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = 100

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.values),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, values, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.values.append(values)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
    
    def get_memory(self):
        return self.states, self.actions, self.probs, self.values, self.rewards, self.dones 

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
            nn.Linear((26+128)*10*9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 161),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, state):
        dist = self.actor_cnn(state)
        dist = T.cat((state, dist), dim = 1) # this will combined them into 156x10x9
        dist = dist.view(dist.size(0),-1) # flatten all dimensions except for the batch dimension (dist.size(0) is the batch, -1 means autodim the rest)
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
            nn.Linear((26+128)*10*9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, state):
        value = self.critic_cnn(state)
        value = T.cat((state, value), dim = 1)
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
        self.win_list = []
        self.entropy_coefficient = 0.001

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
        checkpoint = {'actor_state': self.actor.state_dict(), 'critic_state': self.critic.state_dict(), 'actor_optimizer': self.actor.optimizer.state_dict(), 
                      'critic_optimizer': self.critic.optimizer.state_dict(), 'win_list':self.win_list}
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

    def remember(self, state, action, probs, values, reward, done):
        self.memory.store_memory(state, action, probs, values, reward, done)
        
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

    def learn(self, batch_size): # we can definitely split this stuff into multiple batches and then train on the batches by themselves using multiprocessing
        self.memory.batch_size = batch_size
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(DEVICE)

            values = T.tensor(values).to(DEVICE)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(DEVICE)
                old_probs = T.tensor(old_prob_arr[batch]).to(DEVICE)
                actions = T.tensor(action_arr[batch]).to(DEVICE)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                entropy = dist.entropy().mean()
                total_loss = actor_loss + 0.5*critic_loss - self.entropy_coefficient*entropy
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        self.memory.clear_memory()
        return actor_loss, critic_loss
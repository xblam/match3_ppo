from gym_match3.envs.match3_env import Match3Env
from PPO import *
import time
import argparse

def display_model(num_episodes=10, model_id=int):
    env = Match3Env()
    env.render()
    agent = Agent()
    agent.load_model(model_id)
    print("LOADING PREVIOUS MODEL")
    n_steps = 0
    current_level = 0
    game_won = []

    for current_episode in range(num_episodes):
        n_steps = 0
        obs, infos = env.sample()
        done = False

        while not done:
            mon_hp = env.get_mon_hp()
            player_hp = env.get_player_hp()
            action, _, _ = agent.choose_action(obs, infos)
            env.render(action)
            obs, reward, done, infos = env.step(action)
            n_steps += 1
            print('current level', current_level, '\nepisode', current_episode,'\ntime_steps', n_steps, '\nplayer hp', player_hp, '\nmonster_hp', mon_hp)

        if reward['game'] > 0: 
            print('MONSTER KILLED, ROUND WON')
            game_won.append(1)
            current_level += 1
        else: 
            current_level = 0
            print('ROUND HAS BEEN LOST')
            game_won.append(0)
        print('win rate', sum(game_won[-500:])/min(500,current_episode+1))
        time.sleep(3)

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=int)
    args = parser.parse_args()

    display_model(100, args.model)
   

if __name__ == '__main__':
    main()

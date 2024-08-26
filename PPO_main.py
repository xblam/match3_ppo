from torch.distributions.categorical import Categorical
from gym_match3.envs.match3_env import Match3Env
import wandb
import cProfile, pstats
from PPO import *
import cProfile, pstats
import argparse

def run(num_episodes=1000, log=True, load=False, model_id=22):
    moves_dict = dict()
    env = Match3Env()
    agent = Agent()
    current_level = 0
    game_won = []
    learn_iters = 0

    if load: 
        agent.load_model(model_id)
        print("LOADING PREVIOUS MODEL")
    run_name =  f'heavy {agent.run_id}-{model_id}' if load else f'heavy {agent.run_id}'

    if log: wandb.init(project="match3_easy_ppo", name=str(run_name))

    for current_episode in range(num_episodes): 
        # set episode damange and steps to 0
        episode_damage = 0
        n_steps = 0
        
        # get new game board, observations, and valid moves
        obs, infos = env.sample()
        done = False
        while not done:
            # get the health of the monster and the player before we reset the game
            mon_hp = env.get_mon_hp()
            player_hp = env.get_player_hp()

            # choose an action masked by the valid_actions list and take the action, and record the results
            action, prob, val = agent.choose_action(obs, infos)
            new_obs, reward, done, infos = env.step(action) 

            moves_dict[action] = moves_dict.get(action, 0) + 1

            # increment the steps, and add to total damage. Save all these values to memory
            n_steps += 1
            damage = reward['power_damage_on_monster'] + reward['match_damage_on_monster'] 
            episode_damage += damage
            agent.remember(obs, action, prob, val, damage, done)

            # update the observations and print out the figures and stats for troubleshooting
            obs = new_obs
            print('run id', agent.run_id, '\nepisode', current_episode, '\nscore', episode_damage, '\ntime_steps', n_steps, '\nlearning_steps', learn_iters, '\nreward', reward, '\nplayer hp', player_hp, '\nmonster_hp', mon_hp)

        # when the game is over, we will train the model, need to give it the end game reward so it can factor it in when updating model
        actor_loss, critic_loss = agent.learn(reward['game'])
        learn_iters += 1
        # check the game reward to see what level we are on and update the win rate. we will also not save the state of the model unless it wins a game
        if reward['game'] > 0: 
            print('MONSTER KILLED, MOVING ON TO NEXT LEVEL')
            current_level += 1
            game_won.append(1)
            agent.save_model()
        else: 
            current_level = 0
            game_won.append(0)
        win_rate = sum(game_won[-500:])/(min(current_episode+1, 500))
        agent.win_list = game_won

        
        # log all of the information with wandb
        if log: wandb.log({"episode_damage":episode_damage, "episode":current_episode, 'game reward':reward['game'], 'total reward':reward['game']+
                           episode_damage, 'monster remaining hp' : mon_hp, 'player remaining hp':player_hp, 'actor loss': actor_loss, 'critic loss':critic_loss, 'current level':current_level, 'win rate':win_rate})

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episodes', type=int)
    parser.add_argument('-log', '--log', action='store_true')
    parser.add_argument('-m', '--model', type=int)
    args = parser.parse_args()

    load_model = False
    if args.model: load_model = True
    
    run(args.episodes, args.log, load_model, args.model)
   

if __name__ == '__main__':

    profiler = cProfile.Profile()
    profiler.enable()

    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile_results.prof')



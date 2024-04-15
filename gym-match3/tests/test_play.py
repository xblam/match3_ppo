from gym_match3.envs import Match3Env
from gym_match3.envs.levels import LEVELS #  default levels
from gym_match3.envs.levels import Match3Levels, Level

# create an instance with extended levels
custom_m3_levels = Match3Levels(levels=LEVELS) 
env = Match3Env(levels=custom_m3_levels) 

env.render()
print(env.__get_available_actions())

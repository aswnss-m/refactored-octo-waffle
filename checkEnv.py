from stable_baselines3.common.env_checker import check_env
from CustomEnv import TraciEnv

env = TraciEnv()
check_env(env)
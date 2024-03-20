from stable_baselines3.common.env_checker import check_env
from traciEnv import TraciEnv
# from CustomEnv import TraciEnv
env = TraciEnv()
print(f"Environment created")
check_env(env)
print(f"Environment checked")
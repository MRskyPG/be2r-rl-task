from stable_baselines3 import PPO
from envs.barkour_env import BarkourEnv

env = BarkourEnv()

model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    n_epochs=20,
    verbose=1
)

model.learn(total_timesteps=50_000)
model.save("ppo_barkour")

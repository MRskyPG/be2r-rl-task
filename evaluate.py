import numpy as np
import pickle
from stable_baselines3 import PPO
from envs.barkour_env import BarkourEnv


def run_test(env, model, episodes=100, max_steps=2000):
    logs = {
        "reward_ep": [],
        "reward_steps": [],
        "roll": [],
        "pitch": [],
        "height": [],
        "vel": [],
        "energy": [],
        "ep_len": []
    }

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False

        ep_reward = 0.0
        step_rewards = []
        roll, pitch, height, vel, energy = [], [], [], [], []

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, r, done, _, info = env.step(action)

            ep_reward += r
            step_rewards.append(r)

            roll.append(info["roll"])
            pitch.append(info["pitch"])
            height.append(info["height"])
            vel.append(info["vel"])
            energy.append(info["energy"])

            if done:
                break


        logs["reward_ep"].append(ep_reward)
        logs["reward_steps"].append(step_rewards)
        logs["roll"].append(roll)
        logs["pitch"].append(pitch)
        logs["height"].append(height)
        logs["vel"].append(vel)
        logs["energy"].append(np.sum(energy))
        logs["ep_len"].append(len(step_rewards))

    return logs


model = PPO.load("ppo_barkour")
results = {}

for mass in [0.5, 1.0, 2.0]:
    env = BarkourEnv()
    env.model.body_mass[:] *= mass

    results[mass] = run_test(env, model)

    print(
        f"Масса *{mass}: "
        f"Средняя награда={np.mean(results[mass]['reward_ep']):.3f}, "
        f"Длина эпизода={np.mean(results[mass]['ep_len']):.1f}"
    )

with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

print("Сохранено в файл results.pkl")

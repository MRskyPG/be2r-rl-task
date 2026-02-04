import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from envs.barkour_env import BarkourEnv


MODEL_PATH = "ppo_barkour"
MASS_SCALE = 1.0
DT_SLEEP = 0.01


def main():
    env = BarkourEnv()
    env.model.body_mass[:] *= MASS_SCALE

    model = PPO.load(MODEL_PATH)

    obs, _ = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            viewer.sync()
            time.sleep(DT_SLEEP)

            if done:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()

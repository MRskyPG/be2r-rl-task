"""
Barkour Robot Environment for RL

This project uses robot model files from the Barkour VB developed by
Google DeepMind, including the robot XML configuration and STL mesh files.

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/google_barkour_vb
Copyright: 2023 DeepMind Technologies Limited
License: Apache License 2.0

All other modules (this barkour_env, train, eveluate, plot, play) are developed by Mikhail Rogalsky, 2026
"""
import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation


class BarkourEnv(gym.Env):
    def __init__(self, model_path="scene.xml", settling_steps=50):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.max_episode_steps = 2000

        # Действующее поле действий по количеству актуаторов
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )

        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Вектор: [x, y, z, кватернион(4), 12 суставов (abd, hip, knee) по ногам 1-4]. Взят по аналогии из scene.xml
        self.home_qpos = np.array([
            0.0, 0.0, 0.28,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0,
            0.0, 0.5, 1.0
        ], dtype=np.float64)

        self.steps = 0
        self.target_height = 0.28
        self.target_vel = 0.0
        self.settling_steps = settling_steps

    def _quat_to_euler(self, quat):
        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return rotation.as_euler('xyz')

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)

        n = min(len(self.home_qpos), self.model.nq)
        self.data.qpos[:n] = self.home_qpos[:n]
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        zero_action = np.zeros(self.model.nu, dtype=np.float32)
        for _ in range(self.settling_steps):
            self.data.ctrl[:] = zero_action
            mujoco.mj_step(self.model, self.data)

        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:] = np.clip(action, -1.0, 1.0)
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._reward(action)
        done = self._done()
        info = self._diagnostics(action)

        if done and self.steps < self.max_episode_steps:
            reward -= 5.0  # штраф за раннее падение

        self.steps += 1
        return obs, reward, done, False, info

    def _get_obs(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def _reward(self, action):
        # ориентация
        quat = self.data.qpos[3:7]
        roll, pitch, _ = self._quat_to_euler(quat)

        height = float(self.data.qpos[2])
        vel = float(self.data.qvel[0])

        # штраф, зависящий от величины ошибки удержания высоты
        height_err = abs(height - self.target_height)
        height_reward = -5.0 * height_err

        # штрафы по тангажу и крену, скорости и энергии
        orientation_penalty = -5.0 * (roll ** 2 + pitch ** 2)
        vel_penalty = -2.0 * (vel ** 2)
        energy_penalty = 0.01 * float(np.sum(np.square(self.data.ctrl)))

        joints_qpos = self.data.qpos[7:7 + (self.model.nq - 7)]
        ref_joints = self.home_qpos[7:7 + len(joints_qpos)]
        pose_err = np.linalg.norm(joints_qpos - ref_joints)
        imitation_reward = -5.0 * pose_err  # штраф, зависящий от разницы начальной и текущей позы

        survival_bonus = 5.1 if height > 0.25 else 0.0

        total_reward = (height_reward + orientation_penalty + vel_penalty
                        - energy_penalty + imitation_reward + survival_bonus)

        return float(total_reward)

    def _done(self):
        quat = self.data.qpos[3:7]
        roll, pitch, _ = self._quat_to_euler(quat)
        height = float(self.data.qpos[2])

        # Завершение эпизода при условиях:
        height_failed = height < 0.15
        roll_failed = abs(roll) > 0.8
        pitch_failed = abs(pitch) > 0.8
        timeout = self.steps >= self.max_episode_steps

        return bool(height_failed or roll_failed or pitch_failed or timeout)

    def _diagnostics(self, action):
        quat = self.data.qpos[3:7]
        roll, pitch, _ = self._quat_to_euler(quat)

        energy = float(np.sum(np.square(action)))
        return {
            "roll": float(roll),
            "pitch": float(pitch),
            "height": float(self.data.qpos[2]),
            "vel": float(self.data.qvel[0]) if self.data.qvel.size > 0 else 0.0,
            "energy": energy,
            "steps": int(self.steps)
        }

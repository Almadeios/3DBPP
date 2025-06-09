# environments/packing_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os
import pybullet as p
import numpy as np
from algorithms.dqn_rainbow.pointnet import PointNet
from utils.data_loader import get_episode_data

class PackingEnv(gym.Env):
    def __init__(self, args, shared_interface):
        self.interface = shared_interface
        super().__init__()
        self.args = args
        self.max_objects = args.max_objects
        self.dataset = get_episode_data(args, shared_interface)  # ← ahora pasa la interface
        self.current_episode = -1
        self.objects = []
        self.shape_obs_dim = 128 + 100  # ← actualiza dimensión si PointNet da 128
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.shape_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(100)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_episode += 1
        if self.current_episode >= len(self.dataset):
            self.current_episode = 0

        self.sequence = self.dataset[self.current_episode]
        self.current_index = 0
        self.failed_attempts = 0
        self.interface.reset_scene()
        self.interface.placed_objects = []
        self.interface.objects = []

        obs = self._get_observation()
        return obs, {}

    def step(self, action_idx):
        assert self.current_index < self.max_objects, "Se superó el número máximo de objetos."

        obj = self.sequence[self.current_index]
        candidates = self._generate_candidates(obj)

        # --- Si no hay candidatos válidos, termina el episodio inmediatamente (Zhao-style) ---
        if len(candidates) == 0:
            print(f"[FIN TEMPRANO] No se encontraron poses válidas para el objeto {self.current_index}.")
            done = True
            obs = self._get_observation()
            info = {
                "placed": False,
                "volume_ratio": self.interface.compute_filled_volume_ratio_voxel()["voxel_ratio"],
                "stability": 0.0,
            }
            return obs, -1.0, done, False, info

        # --- Validación del índice de acción recibido ---
        if action_idx >= len(candidates):
            print(f"[ERROR] Acción {action_idx} fuera de rango válido ({len(candidates)}). Penalización.")
            reward = -1.0
            done = False
            return self._get_observation(), reward, done, False, {
                "placed": False,
                "volume_ratio": self.interface.compute_filled_volume_ratio_voxel()["voxel_ratio"],
                "stability": 0.0,
            }

        # --- Aplicar acción seleccionada ---
        pose = candidates[action_idx]
        prev_vol = self.interface.compute_filled_volume_ratio_voxel()["voxel_ratio"]
        valid, stability = self.interface.try_place_object(obj, pose)

        if valid:
            new_vol = self.interface.compute_filled_volume_ratio_voxel()["voxel_ratio"]
            delta_vol = max(0.0, new_vol - prev_vol)
            reward = 0.6 * delta_vol + 0.3 * stability + 0.1  # bonus fijo por éxito
            self.current_index += 1
            done = (self.current_index >= self.max_objects)
        else:
            print(f"[INVALID] La pose seleccionada no es válida físicamente. Acción ignorada.")
            reward = -0.5
            done = True  # ← importante: termina el episodio si intenta una acción inválida

        obs = self._get_observation()
        info = {
            "placed": valid,
            "volume_ratio": new_vol if valid else prev_vol,
            "stability": stability if valid else 0.0,
        }
        return obs, reward, done, False, info



    def _generate_candidates(self, obj):
        all_candidates = self.interface.generate_action_candidates(
            obj,
            resolutionH=self.args.resolutionH,
            resolutionRot=self.args.resolutionRot
        )
        valid_candidates = []
        for pose in all_candidates:
            if self.interface.validate_pose_in_isolated_env(obj, pose):
                valid_candidates.append(pose)
                if len(valid_candidates) >= 20:
                    break
        self.valid_action_indices = list(range(len(valid_candidates)))
        return valid_candidates


    def _get_observation(self):
        if self.current_index < len(self.sequence):
            shape_feat = self.sequence[self.current_index]['shape_feat']
        else:
            shape_feat = np.zeros(128, dtype=np.float32)

        obj = self.sequence[self.current_index] if self.current_index < len(self.sequence) else None
        candidates = self._generate_candidates(obj) if obj else []
        num_candidates = min(len(candidates), 100)
        candidate_encoding = np.zeros(100, dtype=np.float32)
        candidate_encoding[:num_candidates] = 1.0

        obs = np.concatenate([shape_feat, candidate_encoding]).astype(np.float32)
        return obs

    def compute_volume_efficiency(self):
        if hasattr(self.interface, "compute_filled_volume_ratio_voxel"):
            return self.interface.compute_filled_volume_ratio_voxel()
        else:
            return {"real_ratio": 0.0}
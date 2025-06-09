# utils/interface.py

import pybullet as p
import pybullet_data
import numpy as np
import os
import time
from pathlib import Path
import trimesh
import concurrent.futures
from gymnasium import spaces
from utils.data_loader import get_episode_data

def random_soft_color(alpha=1.0):
    base = np.random.rand(3) * 0.5 + 0.5  # componentes entre 0.5 y 1.0
    return [base[0], base[1], base[2], alpha]

class Interface:
    def __init__(self, args, visual=False):
        self.args = args
        self.visual = visual
        self.max_objects = args.max_objects

        self.current_episode = -1
        self.objects = []
        self.placed_objects = []

        self.failed_attempts = 0
        self.max_failures_per_object = 10

        self.shape_obs_dim = 128 + 100
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.shape_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(100)

        # Constantes físicas y contenedor
        self.container_size = self.args.container_size
        self.container_path = os.path.abspath("container.urdf")
        half_L = self.container_size / 2
        self.inner_min = -half_L + 0.01
        self.inner_max =  half_L - 0.01
        self.inner_size = self.inner_max - self.inner_min

        # Crear conexión física
        if self.visual:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=1. / 240., physicsClientId=self.client)

        self._build_container()



    def _build_container(self):
        container_path = os.path.abspath("container.urdf")
        self.container_id = p.loadURDF(container_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=self.client)

    def reset_scene(self):
        for obj_id in self.objects:
            p.removeBody(obj_id, physicsClientId=self.client)
        self.objects.clear()

    def is_inside_container(self, aabb_min, aabb_max, margin=0.005):
        L = self.container_size
        half_L = L / 2
        return (
            aabb_min[0] >= -half_L - margin and aabb_max[0] <= half_L + margin and
            aabb_min[1] >= -half_L - margin and aabb_max[1] <= half_L + margin and
            aabb_min[2] >= -margin  # permite penetración leve en z
        )

    def decode_action_index(self, index):
        resH = self.args.resolutionH
        resR = self.args.resolutionRot
        x = (index // (resH * resR)) % resH
        y = (index // resR) % resH
        r = index % resR
        step = self.args.container_size / resH
        half_L = self.args.container_size / 2
        position = [
            -half_L + (x + 0.5) * step,
            -half_L + (y + 0.5) * step,
            0.5 * self.args.container_size
        ]
        orientation = [0, 0, r * (2 * np.pi / resR)]
        return position, orientation

    def get_mesh_data(self, obj_id, urdf_path):
        try:
            obj_path = Path(urdf_path).with_suffix(".obj")
            if not obj_path.exists():
                return np.zeros((0, 3))
            mesh = trimesh.load_mesh(str(obj_path), force='mesh')
            if not isinstance(mesh, trimesh.Trimesh):
                return np.zeros((0, 3))
            return mesh.sample(1000)
        except Exception:
            return np.zeros((0, 3))

    def _get_pose(self, body_id):
        return p.getBasePositionAndOrientation(body_id, physicsClientId=self.client)

    def get_mesh_points_in_world(self, obj_id, num_points=1000):
        mesh_pts = self.get_mesh_data(obj_id)
        pos, orn = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        world_pts = np.dot(mesh_pts, rot_matrix.T) + np.array(pos)
        return world_pts

    def compute_filled_volume_ratio(self, resolution=32):
        L = self.container_size
        volume_real_total = 0.0
        for obj_id, urdf_path in self.placed_objects:
            try:
                obj_path = Path(urdf_path).with_suffix(".obj")
                if obj_path.exists():
                    mesh = trimesh.load_mesh(str(obj_path), force='mesh')
                    if isinstance(mesh, trimesh.Trimesh):
                        volume_real_total += mesh.volume
            except Exception:
                continue
        return {"real_ratio": volume_real_total / (L ** 3)}

    def try_place_object(self, obj, pose, max_steps=600):
        urdf_path = obj['urdf']
        position, orientation = pose

        x = np.clip(position[0], self.inner_min, self.inner_max)
        y = np.clip(position[1], self.inner_min, self.inner_max)
        drop_z = np.clip(self.estimate_drop_height(x, y) + 0.05, 0.02, self.container_size)
        position = [x, y, drop_z]
        quat = p.getQuaternionFromEuler(orientation)

        obj_id = p.loadURDF(
            urdf_path,
            basePosition=position,
            baseOrientation=quat,
            useFixedBase=False,
            physicsClientId=self.client
        )
        p.changeVisualShape(obj_id, -1, rgbaColor=random_soft_color(), physicsClientId=self.client)
        p.changeDynamics(obj_id, -1, lateralFriction=1.0, restitution=0.0,
                        spinningFriction=1.0, rollingFriction=0.01, physicsClientId=self.client)

        stable_steps = 0
        threshold = 0.08  # más tolerancia que 0.05
        for _ in range(max_steps):
            p.stepSimulation(physicsClientId=self.client)
            lin_vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=self.client)
            if np.linalg.norm(lin_vel) < threshold and np.linalg.norm(ang_vel) < threshold:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= 5:
                break
            if self.visual:
                time.sleep(1. / 240.)

        aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=self.client)
        if not self.is_inside_container(aabb_min, aabb_max, margin=0.005):
            p.removeBody(obj_id, physicsClientId=self.client)
            return False, 0.0

        for other_id, _ in self.placed_objects:
            contacts = p.getContactPoints(obj_id, other_id, physicsClientId=self.client)
            if contacts:
                p.removeBody(obj_id, physicsClientId=self.client)
                return False, 0.0

        self.objects.append(obj_id)
        self.placed_objects.append((obj_id, urdf_path))
        print(f"[USADO] Posición: {position} | Rotación: {orientation} | AABB: {aabb_min} - {aabb_max}")

        lin_speed = np.linalg.norm(lin_vel)
        ang_speed = np.linalg.norm(ang_vel)
        stability = max(0.0, 1.0 - (lin_speed + ang_speed))
        return True, stability

    def has_valid_action(self, obj, resolutionH, resolutionRot):
        total_actions = resolutionH * resolutionH * resolutionRot
        for index in range(total_actions):
            position, orientation = self.decode_action_index(index)
            success, _ = self.try_place_object(obj, (position, orientation), max_steps=240)
            if success:
                p.removeBody(self.objects.pop())
                self.placed_objects.pop()
                return True
        return False

    def _try_pose(self, obj, pose):
        return self.try_place_object(obj, pose)

    def find_valid_action_parallel(self, obj, resH, resR, max_threads=4):
        spacing = self.inner_size / (resH - 1)
        candidate_poses = [
            ([self.inner_min + x * spacing, self.inner_min + y * spacing, 0.5 * self.container_size], [0, 0, r * (2 * np.pi / resR)])
            for x in range(resH)
            for y in range(resH)
            for r in range(resR)
        ]

        def simulate_pose(pose):
            cid = p.connect(p.DIRECT)
            p.setGravity(0, 0, -9.8, physicsClientId=cid)
            p.loadURDF(self.container_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)
            try:
                urdf_path = obj['urdf']
                pos, orn = pose
                drop_z = self.estimate_drop_height(pos[0], pos[1]) + 0.05
                obj_id = p.loadURDF(urdf_path, basePosition=[pos[0], pos[1], drop_z], baseOrientation=p.getQuaternionFromEuler(orn), useFixedBase=False, physicsClientId=cid)
                for _ in range(120):
                    p.stepSimulation(physicsClientId=cid)
                aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=cid)
                for i in range(3):
                    if aabb_min[i] < 0.01 or aabb_max[i] > self.container_size - 0.01:
                        p.disconnect(cid)
                        return None
                p.disconnect(cid)
                return pose
            except:
                p.disconnect(cid)
                return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [executor.submit(simulate_pose, pose) for pose in candidate_poses]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    return result
        return None

    def get_heightmap(self, grid_size=10):
        hmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        cell_size = self.container_size / grid_size
        for obj_id, _ in self.placed_objects:
            pos, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.client)
            x, y, z = pos
            i = int((x + self.container_size / 2) / cell_size)
            j = int((y + self.container_size / 2) / cell_size)
            if 0 <= i < grid_size and 0 <= j < grid_size:
                hmap[i, j] = max(hmap[i, j], z)
        return hmap.flatten()

    def estimate_drop_height(self, x, y, margin=0.02):
        z_max = 0.0
        for obj_id, _ in self.placed_objects:
            aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=self.client)
            if aabb_min[0] - margin <= x <= aabb_max[0] + margin and aabb_min[1] - margin <= y <= aabb_max[1] + margin:
                z_max = max(z_max, aabb_max[2])
        return z_max

    def generate_action_candidates(self, obj, resolutionH=8, resolutionRot=8):
        """
        Genera acciones candidatas con altura de colocación adaptativa (drop height estimada).
        """
        L = self.container_size
        candidates = []

        step = L / resolutionH
        angles = np.linspace(0, 2 * np.pi, resolutionRot, endpoint=False)

        half_L = L / 2
        for x in np.linspace(-half_L + step / 2, half_L - step / 2, resolutionH):
            for y in np.linspace(-half_L + step / 2, half_L - step / 2, resolutionH):
                z = self.estimate_drop_height(x, y) + 0.05  # margen extra para caída
                for angle in angles:
                    pos = [x, y, z]
                    orn = [0, 0, angle]
                    candidates.append((pos, orn))

        return candidates

    def validate_pose_in_isolated_env(self, obj, pose, max_steps=240, margin=None):
        if margin is None:
            margin = 0.005

        urdf_path = obj['urdf']
        position, orientation = pose

        # Estimar altura de caída basada en XY
        x = np.clip(position[0], -self.container_size / 2 + 0.02, self.container_size / 2 - 0.02)
        y = np.clip(position[1], -self.container_size / 2 + 0.02, self.container_size / 2 - 0.02)
        drop_z = max(0.02, self.estimate_drop_height(x, y) + 0.05)
        clipped_pos = [x, y, drop_z]
        quat = p.getQuaternionFromEuler(orientation)

        cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8, physicsClientId=cid)
        p.setPhysicsEngineParameter(fixedTimeStep=1. / 240., physicsClientId=cid)

        try:
            p.loadURDF(self.container_path, basePosition=[0, 0, 0], useFixedBase=True, physicsClientId=cid)

            obj_id = p.loadURDF(
                urdf_path,
                basePosition=clipped_pos,
                baseOrientation=quat,
                useFixedBase=False,
                physicsClientId=cid
            )
            p.changeVisualShape(obj_id, -1, rgbaColor=random_soft_color(), physicsClientId=cid)

            threshold = 0.08
            stable_steps = 0
            stable_steps_required = 5
            for _ in range(max_steps):
                p.stepSimulation(physicsClientId=cid)
                lin_vel, ang_vel = p.getBaseVelocity(obj_id, physicsClientId=cid)
                if np.linalg.norm(lin_vel) < threshold and np.linalg.norm(ang_vel) < threshold:
                    stable_steps += 1
                else:
                    stable_steps = 0
                if stable_steps >= stable_steps_required:
                    break

            aabb_min, aabb_max = p.getAABB(obj_id, physicsClientId=cid)
            if not self.is_inside_container(aabb_min, aabb_max, margin):
                p.disconnect(cid)
                return False

            p.disconnect(cid)
            return True
        except Exception:
            p.disconnect(cid)
            return False

    def compute_filled_volume_ratio_voxel(self, resolution=32):
        """
        Voxeliza el contenedor y objetos colocados para estimar el volumen ocupado.
        Retorna tanto el ratio voxelizado como el volumen real.
        """
        voxel_size = self.container_size / resolution
        occupied = set()
        real_volume = 0.0

        for obj_id, urdf_path in self.placed_objects:
            mesh = self.get_mesh_data(obj_id, urdf_path)
            if mesh is None or len(mesh) == 0:
                continue
            pos, orn = self._get_pose(obj_id)
            rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            transformed = np.dot(mesh, rot_matrix.T) + pos

            indices = ((transformed + self.container_size / 2) / voxel_size).astype(int)
            for idx in indices:
                if np.all(idx >= 0) and np.all(idx < resolution):
                    occupied.add(tuple(idx))

            # Volumen real desde el archivo .obj
            try:
                obj_path = Path(urdf_path).with_suffix(".obj")
                if obj_path.exists():
                    trimesh_obj = trimesh.load_mesh(str(obj_path), force='mesh')
                    if isinstance(trimesh_obj, trimesh.Trimesh):
                        real_volume += trimesh_obj.volume
            except Exception:
                pass

        voxel_ratio = len(occupied) / (resolution ** 3)
        real_ratio = real_volume / (self.container_size ** 3)

        return {
            "voxel_ratio": voxel_ratio,
            "real_ratio": real_ratio
        }

import sys
import os
import torch
import numpy as np
import csv
from arguments import get_args
from algorithms.dqn_rainbow.agent import RainbowAgent
from algorithms.dqn_rainbow.memory import Memory

# === Silenciar salidas C++ de PyBullet (como 'argv[0]=') ===
class CppOutputSuppressor:
    def __enter__(self):
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.stdout_fd = sys.__stdout__.fileno()
        self.stderr_fd = sys.__stderr__.fileno()
        self.saved_stdout = os.dup(self.stdout_fd)
        self.saved_stderr = os.dup(self.stderr_fd)
        os.dup2(self.devnull, self.stdout_fd)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout, self.stdout_fd)
        os.dup2(self.saved_stderr, self.stderr_fd)
        os.close(self.devnull)
        os.close(self.saved_stdout)
        os.close(self.saved_stderr)

# Solo durante importación de PyBullet
with CppOutputSuppressor():
    import pybullet
    import pybullet_data
    from pybullet_utils import bullet_client

from utils.interface import Interface
from environments.packing_env import PackingEnv

import gymnasium as gym
import environments  # registra Physics-v0

def evaluate(env, agent, episode, episodes=5):
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            valid_actions = env.valid_action_indices
            if not valid_actions:
                break
            action = agent.select_action(obs, valid_actions=valid_actions, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    avg_reward = total_reward / episodes
    print(f"[Evaluación] Recompensa promedio: {avg_reward:.2f}")

    # Guardar en CSV
    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/eval_rewards.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["episode", "avg_reward"])
        writer.writerow([episode, round(avg_reward, 4)])

    return avg_reward

def train():
    args = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Usando dispositivo: {device}")

    shared_interface = Interface(args, visual=args.visual)
    env = PackingEnv(args, shared_interface)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = RainbowAgent(obs_dim, action_dim, args, device)
    memory = Memory(capacity=args.memory_capacity, alpha=args.per_alpha)

    os.makedirs(args.model_dir, exist_ok=True)
    final_model_path = os.path.join(args.model_dir, "final_model.pt")
    if os.path.exists(final_model_path):
        agent.policy_net.load_state_dict(torch.load(final_model_path, map_location=device))
        agent.policy_net.to(device)
        agent.policy_net.eval()
        print(f"[CARGADO] Modelo previo cargado desde: {final_model_path}")
    else:
        print("[ENTRENAMIENTO NUEVO] No se encontró modelo previo. Entrenando desde cero.")

    steps = 0
    early_stops = 0

    for episode in range(1, args.max_episodes + 1):
        obs, _ = env.reset()

        if hasattr(env.unwrapped.interface, "placed_objects"):
            env.unwrapped.interface.placed_objects = []

        done = False
        total_reward = 0
        paso_ep = 0

        print(f"\n[INICIO] Episodio {episode} | Shape obs: {obs.shape}")

        while not done:
            steps += 1
            paso_ep += 1
            valid_actions = env.valid_action_indices
            if not valid_actions:
                print("[AVISO] No hay acciones válidas.")
                break
            action = agent.select_action(obs, valid_actions=valid_actions)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32)

            reward = np.clip(reward, -1.0, 1.0)

            # Penalización adicional si el episodio termina sin colocar objetos
            if done and len(env.unwrapped.interface.placed_objects) == 0:
                print("[CASTIGO] Episodio terminó sin colocar ningún objeto. Penalizando...")
                reward = -10.0  # penalización fuerte por no lograr ningún objetivo
            elif done and reward <= -1.0:
                print(f"[CASTIGO] Acción inválida al final. Penalización leve aplicada.")
                reward = -2.0

            memory.add(1.0, (obs_tensor, action, reward, next_obs_tensor, done))

            if steps > args.learn_start:
                batch, indices, weights = memory.sample(args.batch_size, beta=args.per_beta)
                agent.learn(batch, indices, weights, memory)

            obs = next_obs
            total_reward += reward

            if steps % args.target_update == 0:
                agent.update_target()

            if paso_ep % 5 == 0:
                print(f"  [Paso {paso_ep:03d}] Acción: {action} | Rew parcial: {total_reward:.2f}")

        if paso_ep == 1 and 'reward' in locals() and reward <= -1.0:
            early_stops += 1
            print("[INFO] Episodio terminado por falta de acciones válidas.")

        num_colocados = len(env.unwrapped.interface.placed_objects)
        proporcion = env.unwrapped.compute_volume_efficiency()
        porc_vol = proporcion['real_ratio'] * 100

        print(f"[Episodio {episode}] Recompensa total: {total_reward:.2f} | Pasos: {paso_ep}")
        print(f"  Objetos colocados: {num_colocados} | Volumen ocupado: {porc_vol:.2f}%")

        # Guardar métricas del entrenamiento por episodio
        os.makedirs("logs", exist_ok=True)
        csv_path = "logs/train_stats.csv"
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["episode", "total_reward", "steps", "objects_placed", "volume_pct"])
            writer.writerow([episode, round(total_reward, 4), paso_ep, num_colocados, round(porc_vol, 2)])

        # Evaluación periódica
        if episode % args.evaluation_episodes_training == 0:
            evaluate(env, agent, episode, episodes=args.evaluation_episodes)

    torch.save(agent.policy_net.state_dict(), final_model_path)
    print(f"\n[FINALIZADO] Modelo completo guardado como: {final_model_path}")
    print(f"[RESUMEN] Total de episodios sin acciones válidas: {early_stops}")

if __name__ == '__main__':
    train()

import os
import torch
import numpy as np
import csv
from arguments import get_args
from algorithms.dqn_rainbow.agent import RainbowAgent
from utils.interface import Interface
from environments.packing_env import PackingEnv

def main():
    args = get_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo: {device}")

    args.visual = True  # si deseas inspección visual
    shared_interface = Interface(args, visual=args.visual)
    env = PackingEnv(args, shared_interface)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = RainbowAgent(obs_dim, action_dim, args, device)

    model_path = os.path.join(args.model_dir, "final_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {model_path}")
    
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=device))
    agent.policy_net.eval()

    os.makedirs("results", exist_ok=True)
    csv_path = "results/evaluation_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "objects_placed", "volume_pct"])

        for ep in range(1, args.evaluation_episodes + 1):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            done = False
            while not done:
                valid_actions = env.valid_action_indices
                if not valid_actions:
                    print(f"[{ep}] No hay acciones válidas. Terminando.")
                    break
                action = agent.select_action(obs, valid_actions=valid_actions, evaluate=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            
            num_colocados = len(env.unwrapped.interface.placed_objects)
            proporciones = env.unwrapped.compute_volume_efficiency()
            porc_vol = proporciones.get("real_ratio", 0.0) * 100
            print(f"[{ep}] Reward: {total_reward:.2f} | Pasos: {steps} | Objetos: {num_colocados} | Vol: {porc_vol:.2f}%")
            writer.writerow([ep, round(total_reward, 4), steps, num_colocados, round(porc_vol, 2)])

if __name__ == "__main__":
    main()

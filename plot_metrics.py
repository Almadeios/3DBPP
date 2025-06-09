# plot_metrics.py

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metrics():
    if not os.path.exists("logs/train_stats.csv"):
        print("No se encontró logs/train_stats.csv")
        return
    if not os.path.exists("logs/eval_rewards.csv"):
        print("No se encontró logs/eval_rewards.csv")
        return

    train_df = pd.read_csv("logs/train_stats.csv")
    eval_df = pd.read_csv("logs/eval_rewards.csv")

    fig1, ax1 = plt.subplots()
    ax1.plot(train_df["episode"], train_df["total_reward"], label="Recompensa total")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa")
    ax1.set_title("Recompensa por episodio")
    ax1.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/recompensa_total.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(train_df["episode"], train_df["volume_pct"], label="% Volumen ocupado")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Volumen (%)")
    ax2.set_title("Volumen ocupado por episodio")
    ax2.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/volumen_ocupado.png")

    fig3, ax3 = plt.subplots()
    ax3.plot(eval_df["episode"], eval_df["avg_reward"], 'o-', label="Recompensa promedio (eval)")
    ax3.set_xlabel("Episodio")
    ax3.set_ylabel("Recompensa promedio")
    ax3.set_title("Evaluación periódica")
    ax3.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("logs/eval_rewards.png")

    print("Gráficos guardados en la carpeta logs/")

if __name__ == "__main__":
    plot_metrics()

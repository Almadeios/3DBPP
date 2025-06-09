import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # Dataset y entorno
    parser.add_argument('--dataset', type=str, default='blockout')
    parser.add_argument('--test_name', type=str, default='test_sequence')
    parser.add_argument('--envName', type=str, default='Physics-v0')
    parser.add_argument('--max_objects', type=int, default=25)

    # Resoluciones
    parser.add_argument('--resolutionRot', type=int, default=6)
    parser.add_argument('--resolutionH', type=int, default=7)

    # Hiperparámetros de RL
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=int, default=25000)

    parser.add_argument('--memory_capacity', type=int, default=100000)
    parser.add_argument('--per_alpha', type=float, default=0.6)
    parser.add_argument('--per_beta', type=float, default=0.4)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learn_start', type=int, default=1000)
    parser.add_argument('--target_update', type=int, default=500)

    # Evaluación
    parser.add_argument('--max_episodes', type=int, default=2000)
    parser.add_argument('--evaluation_episodes_training', type=int, default=100)
    parser.add_argument('--evaluation_episodes', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=100)

    # Guardado
    parser.add_argument('--model_dir', type=str, default='models')

    # Visualización y física
    parser.add_argument('--custom', type=str, default='longrun_v1')
    parser.add_argument('--visual', action='store_true')
    parser.add_argument('--container_size', type=float, default=0.32)
    parser.add_argument("--validation_margin", type=float, default=0.01, help="Margen de validación AABB")
    # GPU
    parser.add_argument('--device', type=int, default=0)

    return parser.parse_args()

# 3D Bin Packing with DQN Rainbow

Este proyecto implementa una solución al problema de empaquetado 3D (3DBPP) en línea utilizando aprendizaje por refuerzo profundo con DQN Rainbow. Usa datos reales de objetos `.obj` y nubes de puntos, con validación de colisiones y observaciones físicas.

---

## 📦 Estructura

3DBPP/
├── datasets/blockout/ # Dataset original con test_sequence.pt,id2shape.pt, pointCloud/, shape_vhacd/
├── environments/physics_env.py # Entorno Gymnasium personalizado
├── algorithms/dqn_rainbow/ # agent, memory, pointnet, trainer
├── utils/visualization.py # Visualización y métricas
├── main.py # Entrenamiento
├── evaluate.py # Evaluación de modelos
├── arguments.py # Hiperparámetros y configuración
└── requirements.txt

## Entrenamiento
python train.py --custom full_training --max_objects 30 --max_episodes 1000 --visual --save_interval 100 --resolutionH 8 --resolutionRot 8
python train.py --custom full_training --max_objects 30 --max_episodes 1000 --save_interval 100 --resolutionH 8 --resolutionRot 8

python plot_metrics.py


## Evaluación
python evaluate.py --max_objects 30 --max_episodes 1000 --save_interval 100 --resolutionH 8 --resolutionRot 8


## Requisitos

## ctrl+shift+P -> Python: Select Interpreter 3.11.0
## .\.venv\Scripts\activate
## pip install -r requirements.txt

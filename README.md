# 3D Bin Packing with DQN Rainbow

Este proyecto implementa una solución al problema de empaquetado 3D (3DBPP) en línea utilizando aprendizaje por refuerzo profundo con DQN Rainbow. Usa datos reales de objetos `.obj` y nubes de puntos, con validación de colisiones y observaciones físicas.

---

## 📦 Estructura
    3DBPP/  
    datasets/blockout/ # Dataset original con test_sequence.pt,id2shape.pt, pointCloud/, shape_vhacd/
    environments/packing_env.py # Entorno Gymnasium personalizado
    algorithms/dqn_rainbow/ # agent, memory, pointnet
    utils/ # data_loader, interface, sequence_loader, shape_features
    train.py
    evaluate.py 
    arguments.py 
    requirements.txt
    plot_metrics.py

## Entrenamiento
    python train.py --custom full_training --max_objects 30 --max_episodes 1000 --visual --save_interval 100 --resolutionH 8 --resolutionRot 8
    python train.py --custom full_training --max_objects 30 --max_episodes 1000 --save_interval 100 --resolutionH 8 --resolutionRot 8

## Metricas
    python plot_metrics.py

## Evaluación
python evaluate.py --max_objects 30 --max_episodes 1000 --save_interval 100 --resolutionH 8 --resolutionRot 8

## Requisitos
    Microsoft Visual C++ 14.0 o superior
    https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version
    "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    python -m pip install --upgrade pip
    https://www.python.org/downloads/release/python-3110/
    ctrl+shift+P -> Python: Select Interpreter -> Create Virtual Enviroment -> venv -> Python 3.11.0
    .\.venv\Scripts\activate
    pip install -r requirements.txt

# utils/data_loader.py

import torch
import os
import numpy as np
from algorithms.dqn_rainbow.pointnet import PointNet

def get_episode_data(args, interface):
    test_seq_path = f"datasets/blockout/{args.test_name}.pt"
    id2shape_path = "datasets/blockout/id2shape.pt"
    shape_dir = "datasets/blockout/shape_vhacd"

    print(f"[DEBUG] Cargando episodios desde: {test_seq_path}")
    print(f"[DEBUG] Cargando id2shape desde: {id2shape_path}")

    if not os.path.exists(test_seq_path):
        raise FileNotFoundError(f"No se encontró el archivo de episodios: {test_seq_path}")
    if not os.path.exists(id2shape_path):
        raise FileNotFoundError(f"No se encontró el archivo id2shape: {id2shape_path}")

    with torch.serialization.safe_globals([np.core.multiarray.scalar]):
        raw_data = torch.load(test_seq_path, weights_only=False)
        id2shape = torch.load(id2shape_path, weights_only=False)

    print(f"[DEBUG] Total de episodios cargados: {len(raw_data)}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    pointnet = PointNet().to(device).eval()

    dataset = []
    for i, episode in enumerate(raw_data[:args.max_episodes]):
        processed_episode = []
        for obj_id in episode[:args.max_objects]:
            filename = id2shape.get(obj_id)
            if filename is None:
                raise ValueError(f"[ERROR] ID {obj_id} no encontrado en id2shape.")
            urdf_path = os.path.join(shape_dir, filename.replace(".obj", ".urdf"))

            mesh = interface.get_mesh_data(obj_id, urdf_path)
            if mesh is None or len(mesh) == 0:
                shape_feat = np.zeros(128, dtype=np.float32)
            else:
                points = torch.tensor(mesh[:1024], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = pointnet(points)
                shape_feat = feat.squeeze(0).cpu().numpy()

            processed_episode.append({
                "urdf": urdf_path,
                "shape_feat": shape_feat.astype(np.float32)
            })
        dataset.append(processed_episode)

    return dataset

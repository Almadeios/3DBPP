import trimesh
import numpy as np
from pathlib import Path

def extract_shape_features(urdf_path, num_points=1024):
    """
    Extrae características geométricas simples de la malla asociada a un URDF.
    Estas no son PointNet pero sirven como aproximación informativa.

    Retorna un vector de 164 floats:
    - 100 coordenadas z de puntos de muestreo (shape descriptor)
    - 1 volumen
    - 3 dimensiones del bounding box
    - 60 bins del histograma de alturas
    """
    obj_path = Path(urdf_path).with_suffix(".obj")
    if not obj_path.exists():
        return np.zeros(164, dtype=np.float32)

    try:
        mesh = trimesh.load_mesh(str(obj_path), force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            return np.zeros(164, dtype=np.float32)

        # Muestreo de puntos
        pts = mesh.sample(num_points)
        z_coords = pts[:, 2]
        z_padded = np.pad(z_coords, (0, max(0, 100 - len(z_coords))), mode='constant')[:100]

        # Volumen y bounding box
        volume = mesh.volume
        bbox = mesh.bounding_box.extents  # (x, y, z)

        # Histograma de alturas
        hist, _ = np.histogram(z_coords, bins=60, range=(0, 1.0))
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-8)

        feat = np.concatenate([
            z_padded.astype(np.float32),
            [volume],
            bbox.astype(np.float32),
            hist
        ])
        return feat  # 100 + 1 + 3 + 60 = 164
    except Exception:
        return np.zeros(164, dtype=np.float32)

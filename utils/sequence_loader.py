# utils/sequence_loader.py
import torch
import os
import numpy as np

def load_test_sequence(test_name):
    """
    Carga la secuencia de prueba desde un archivo .pt en datasets/blockout.
    """
    path = os.path.join("datasets", "blockout", "test_sequence.pt")
    return torch.load(path, weights_only=False)

def load_id2shape(test_name):
    path = os.path.join("datasets", "blockout", "id2shape.pt")
    data = torch.load(path, weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"[load_id2shape] Se esperaba un diccionario, pero se obtuvo: {type(data)}")
    return data



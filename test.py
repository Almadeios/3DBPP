import pybullet as p
import time
from arguments import get_args
from utils.interface import Interface

if __name__ == '__main__':
    args = get_args()
    args.visual = True
    interface = Interface(args, visual=True)

    obj = {
        'urdf': 'datasets/blockout/shape_vhacd/1_1B_0.urdf'
    }

    resolutionH = 6
    resolutionRot = 6
    L = args.container_size
    step = L / resolutionH

    candidates = []
    angles = [i * (2 * 3.14159 / resolutionRot) for i in range(resolutionRot)]

    for x in [step / 2 + i * step for i in range(resolutionH)]:
        for y in [step / 2 + i * step for i in range(resolutionH)]:
            z = interface.estimate_drop_height(x, y) + 0.15  # prueba con +0.15 o incluso 0.2 si sigue tocando
            for angle in angles:
                pos = [x, y, z]
                orn = [0, 0, angle]
                candidates.append((pos, orn))

    print(f"[INFO] Candidatos generados: {len(candidates)}")

    valid_count = 0
    for i, (pos, orn) in enumerate(candidates[:50]):  # Prueba 10 primeros
        is_valid, is_stable = interface.try_place_object(obj, (pos, orn))
        if is_valid and is_stable:
            print(f"[VALIDO] Pose {i}: Pos {pos}, Orn {orn}")
            quat = p.getQuaternionFromEuler(orn)
            obj_id = p.loadURDF(obj['urdf'], basePosition=pos, baseOrientation=quat, useFixedBase=False, physicsClientId=interface.client)
            p.changeVisualShape(obj_id, -1, rgbaColor=[0, 1, 0, 1], physicsClientId=interface.client)

            for _ in range(240):
                p.stepSimulation()
                time.sleep(1. / 240.)

            p.removeBody(obj_id)
            valid_count += 1
        else:
            print(f"[INVALIDO] Pose {i}: Pos {pos}, Orn {orn}")

    print(f"[RESUMEN] Total candidatos válidos mostrados: {valid_count}")

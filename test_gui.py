import pybullet as p
import pybullet_data
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -10)

cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
cubeId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

for i in range(1000):
    p.stepSimulation()
    time.sleep(1./240.)
p.disconnect()

import time

import pybullet
import pybullet_data


if __name__ == '__main__':
    client = pybullet.connect(pybullet.GUI)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0)

    pybullet.setGravity(0, 0, -9.8)
    pybullet.setRealTimeSimulation(1)

    # 载入urdf格式是场景
    pybullet.loadURDF("plane100.urdf", useMaximalCoordinates=True)
    # 载入urdf格式的机器人
    r_ind = pybullet.loadURDF('cassie_w_gen3.urdf', (3, 1, 1), pybullet.getQuaternionFromEuler([0, 0, 0.785]))

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
    while True:
        time.sleep(1. / 240.)



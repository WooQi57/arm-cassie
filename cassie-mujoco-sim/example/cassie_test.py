import gym
from cassie import CassieEnv
import time
import numpy as np
t = time.monotonic()

cassie = CassieEnv()
for epi in range(200):
    observation = cassie.reset()
    for i in range(50):
        print('No.', i)
        cassie.render()
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
        t = time.monotonic()            

        action=np.array([0]*10)
        observation, reward, done, info = cassie.step(action)
        # input('press ENTER')

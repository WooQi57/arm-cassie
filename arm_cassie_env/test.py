from operator import mod
from torch import rand
import gym
from cassie_env.cassieRLEnvStable import cassieRLEnvStable
import time
import random
import numpy as np
t = time.monotonic()

cassie = cassieRLEnvStable(visual=True)
for epi in range(50):
    observation = cassie.reset_test()
    action=np.array([0]*16,dtype='float')
    action[10:]=[random.uniform(-1.5,1.5) for x in range(6)]
    # action[10]=random.uniform(-1.5,1.5)
    # action[11]=-1.57
    # action[10:]=[0,-1.047,-2.618,0,0,0]
    action[10:]=[0,-0.5236,-2.0944,0,0,0]
    # action[10:]=[0,0,0,0,0,0]
    for i in range(150):
        cassie.render()
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
        t = time.monotonic()
        # if i == 0:
        #     input("press any key")

        ## test PD controller
        # print('*'*10)
        # error = [action[10+k]-cassie.sim.qpos()[35+k] for k in range(6)]            
        # print(np.linalg.norm(error))
        print(cassie.sim.xpos('ee'))
        observation, reward, done, info = cassie.step_test(action)
    

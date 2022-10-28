from operator import mod
from torch import rand
import gym
from cassie_env.cassieRLEnvStable import cassieRLEnvStable
import time
import random
import numpy as np
import ikpy.chain as ik

t = time.monotonic()
# arm_chain = ik.Chain.from_urdf_file("./cassie_env/arm/kinova_gen3.urdf")
cassie = cassieRLEnvStable(visual=True,setforce=[0,0,0,0,0,0])
cassie.test=True
# target =[-0.16745495,  0.00135078 , 1.00930341]
target =[0.27681454, 0.0013535 , 0.63987845]
target = [0.28,0,-0.30]

for epi in range(50):

    observation = cassie.reset()
    action=np.array([0]*10,dtype='float')
    cassie.arm_zero=np.array([random.uniform(-1.5,1.5) for x in range(6)])
    for i in range(150):
        cassie.render()
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
        t = time.monotonic()

        # joint=[0]*8
        # joint[5]=1.57
        tori=[1,0,0]
        # realpos=arm_chain.forward_kinematics(joint)

        # target=realpos[:3,3]
        # action[10:]=arm_chain.inverse_kinematics(target,tori, orientation_mode="Z")[1:-1]

        # print(action[10:])
        # action[10:]=joint[1:-1]


        # test PD controller
        # print('*'*10)
        # error = [action[10+k]-cassie.sim.qpos()[35+k] for k in range(6)]            
        # print(np.linalg.norm(error))
        
        observation, reward, done, info = cassie.step(action)
        # input('press ENTER')

from torch import rand
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from cassie import CassieRefEnv
from cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from cassie_env.cassieArm import cassieArm
import matplotlib.pyplot as plt
import random
import numpy as np
import csv


if __name__ == '__main__':
    t = time.monotonic()
    cassie = cassieArm(visual=True,setforce=[0,0,0,0,0,0],fix=True)
    cassie.isplay = True
    # cassie.loadModel('test')
    # cassie.loadModel('flat_model/v1')
    cassie.loadModel('wrenchflatmodel/v1')
    cassie.noisy = False
    cassie.delay = False
    cassie.speed=0.3
    cassie.obs = cassie.reset()

    # print(cassie.sim.xpos('ee')[2])
    h = []
    arm_x = []
    arm_y=[]
    pelvis_x = []

    # with open('armyf.csv','w') as csvfile:
    #     writer = csv.writer(csvfile)
    walkingTimer = 0
    isStop=False

    if cassie.test:
        isStop = True
        cassie.fix = False
        cassie.IK = False

    while True:
        walkingTimer += 1
        # if walkingTimer == 50:
        #     cassie.sim.apply_force([50,0,0,0,0,0],'cassie-pelvis')
        #     print("#"*25)
        # if walkingTimer==57:
        #     cassie.sim.clear_forces()
        if cassie.sim.qpos()[0] > 4.5 and not isStop:  #3.6
            checkPoint = walkingTimer
            isStop = True
            print("stop and move arm")
            cassie.speed=0
            cassie.fix = True
            # cassie.loadModel('standstill')
            # cassie.loadModel('flat_model/step1back0')
            cassie.loadModel('wrenchflatmodel/v1-0')

        if isStop:
            if cassie.sim.qpos()[0] > 5: #walkingTimer-checkPoint == 200:#
                print("moving arm")
                cassie.speed=0
                cassie.fix = False
                cassie.IK = False
                cassie.loadModel('wrenchflatmodel/v0')

                # if walkingTimer-checkPoint > 200:
                #     cassie.sim.set_gripper(True)
        
                    
        obs, rewards, dones, info = cassie.step()
        h.append(cassie.sim.xpos('ee')[2])
        arm_y.append(cassie.sim.xpos('ee')[1])
        arm_x.append(cassie.sim.xpos('ee')[0])
        pelvis_x.append(cassie.sim.qpos()[0])
        # writer.writerow([60/2000*len(arm_y),arm_y[-1]])
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
        t = time.monotonic()
        
        if dones:# or walkingTimer>: 
            walkingTimer = 0
            cassie.speed=1
            cassie.fix = True
            isStop = False
            cassie.IK = False
            # cassie.loadModel('2in1')
            # cassie.loadModel('flat_model/v1')
            cassie.loadModel('wrenchflatmodel/v1')

            obs = cassie.reset()
            plt.plot(h)
            # plt.show()
            h=[]
            plt.plot(arm_x)
            plt.plot(pelvis_x)
            # plt.show()
            arm_x=[]
            pelvis_x=[]
            plt.plot(arm_y)
            # plt.show()
            arm_y=[]

from torch import rand
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
# from cassie import CassieRefEnv
from cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from cassie_env.cassieRLEnvStable import cassieRLEnvStable
import matplotlib.pyplot as plt
import random
import numpy as np


if __name__ == '__main__':
    t = time.monotonic()
    # model = PPO.load("model_saved/ppo_cassie_"+str(512 * 36)+"00")
    # model = PPO.load("./wrenchflatmodel/ppo_cassie-2969600")
    # model = PPO.load("standstill")
    model = PPO.load('wrenchflatmodel/v1')
    # cassie = CassieRefEnv(dynamics_randomization=False)
    cassie = cassieRLEnvStable(visual=True, setforce=[0,0,0,0,0,0])
    cassie.noisy = False
    cassie.delay = False
    obs = cassie.reset()
    # print(len(obs))
    vel = []    
    while True:
        action, _states = model.predict(obs,deterministic=True)
        obs, rewards, dones, info = cassie.step(action)
        # print(len(cassie.sim.qpos()))
        while time.monotonic() - t < 60*0.0005:
            time.sleep(0.0001)
            # cassie.render()
        t = time.monotonic()
        
        # pos_index = np.array([7, 8, 9, 14, 20, 21, 22, 23, 28, 34])
        # qpos=np.array(cassie.sim.qpos())
        # joints=qpos[pos_index].tolist()
        # vel.append(joints)
        
        # vel.append(cassie.sim.qvel()[0])

        vel.append(action)

        if dones: #cassie.qpos[2]<0.6:
            # plt.plot(vel)
            # plt.show()
            # plt.legend()

            vel = []
            # cassie.setforce = 0# random.uniform(0,8)#0-8
            # print("yforce:",cassie.setforce)
            obs = cassie.reset()
            # print(cassie.speed)
            

from torch import rand
from cassie_env.quaternion_function import inverse_quaternion, quaternion2euler, quaternion2mat, quaternion_product
import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from cassie_env.cassieRLEnvStable import cassieRLEnvStable
import random
import numpy as np
import ikpy.chain as ik

       
class cassieArm(cassieRLEnvStable):
    def __init__(self, visual=False, setforce=-1, fix=True):
        super().__init__(visual, setforce,fix)
        # self.legModel = PPO.load("test")        
        self.arm_zero=np.array([0,-0.5236,-2.0944,0,0,0])  # reset arm [0,-1.047,-2.618,0,0,0] 
        self.arm_zero=np.array([0,1.57-0.51,-1.57-0.51,3.14,0,0])  # reset arm [0,-1.047,-2.618,0,0,0] 

        # self.arm_zero=np.array([0,-1.4,1.4,0,0,0])  # light
        self.ik_init=np.array([0,0,1.57-0.51,-1.57-0.51,3.14,0,0,0])
        # self.arm_chain = ik.Chain.from_urdf_file("./cassie_env/arm/wx250s.urdf",base_elements=["wx250s/base_link"])
        self.arm_chain = ik.Chain.from_urdf_file("./cassie_env/arm/kinova_gen3.urdf",base_elements=["EndEffector_Link"])
        self.arm_target =[0.27681454, 0.0013535 , 0.63987845]
        self.arm_height = 1.84 - 0.15
        self.xbuf=[]
        self.IK = False
        self.test = False
        self.IKcnt=0
        self.armtrain=False
        self.chickenhead = False

    def loadModel(self, model):
        self.legModel = PPO.load(model)

    def step(self):
        legAction, _states = self.legModel.predict(self.obs, deterministic = True) 
        if self.test:
            legAction=np.array([0]*10)
        if self.fix == False:
            self.arm_control()
        # print(self.sim.qvel()[0])
        self.obs, rewards, dones, info = super().step(legAction)
        # print([round(x,2) for x in self.sim.arm_wrench()])
        return self.obs, rewards, dones, info

    def reset(self):
        chs=np.array([
            [-1.40375984,  1.85737509, -1.5717031 ,  3.14      , -0.28749207,   -1.40222759],
            [ 1.50926031,  1.85850805, -1.5517621 , -0.05938497,  0.26744237,   -1.57      ],
            [ 0.79022005,  1.81113796, -1.3215943 , -0.78135907, -0.00499692,   -1.57      ],
            [-0.78381726,  1.82780641, -1.22545797,  3.14      ,  0.08832286,  -0.78222309],
            [0,2.0944,-2.0944,0,2*0.5236,0],
            [0,1.57-0.51,-1.57-0.51,0,0,0]
        ])
        # self.arm_setup_pos = random.choice(chs)

        ret = super().reset()
        self.arm_init_pos = self.sim.xpos("ee")
        self.arm_init_quat = self.sim.xquat('ee')
        # self.target_pos = self.sim.xpos("pullknob_link") #[5.992, 0.18159, 1.2049999999999998]
        self.target_pos = self.sim.xpos("box_1") 
        return ret

    def step_simulation(self, action):
        # ApGain = [200, 150,150,50,10,1] 
        # AdGain = [20,20,15,0.5,0.5,0.1]
        
        if self.fix == False:
            # ApGain = [3000, 5000,5000,750,5000,100] 
            ApGain = [3000, 5000,5000,750,5000,100] 
            ApGain = [x/25 for x in ApGain] 
            AdGain = [20,200,30,2,1,1]

            #light
            # ApGain = [100,100,100,10,10,30]
            # AdGain = [20,20,20,1,6,0]

            self.sim.sim_arm(self.arm_setup_pos, ApGain, AdGain)
        super().step_simulation(action)

    def arm_controller(self):
        ee_pos = np.copy(self.target_pos)
        ee_pos[0] -= (0.114+0.111)
        wrist_pos = np.copy(ee_pos)
        wrist_pos[0] -= (0.0615+0.1059)

    def wx250sIK(self):
        height = 0.2
        bodyquat = np.copy(self.sim.qpos()[3:7])
        bodymat = quaternion2mat(bodyquat)
        targetmat = np.linalg.inv(bodymat)
        pos_dif = [self.target_pos[i] - self.sim.qpos()[i] for i in range(3)]

        pos_body2target = np.dot(targetmat,pos_dif)

        wrist_pos = [pos_dif[i] - 0.17415*targetmat[i,0] for i in range(3)]  # wqwqwq 可能不对这个0

        joints=[0]*6
        joints[0] = np.arctan2(pos_body2target[1],pos_body2target[0])
        r = pos_body2target[0]**2 + pos_body2target[1]**2
        s = pos_body2target[2] - 0.225 - 0.03865
        a2 = 0.25495
        a3 = 0.25
        D = (r**2 + s**2 - a2**2 - a3**2)/(2*a2*a3)
        q2 = np.arccos(D)
        joints[1] = np.pi/2 - (np.arctan2(s,r)+np.arctan2(a3*np.sin(q2), a2+a3*D)) - np.arctan(0.2)
        joints[2] = q2 - np.arctan(5)



    def arm_control(self):
        
        if self.test: #hang to test ik
            # if self.IKcnt%200 == 0: 
            #     print("fuck")
            #     self.arm_setup_pos = [random.uniform(-1,1) for x in range(6)]
            # self.IKcnt+=1
            # print([round(x,2) for x in self.sim.qpos()[35:41]])

            tori=[0,0,1]
            target_orientation = np.eye(3)
            self.arm_target =[0.3, 0.0, -0.3 ]
            # self.arm_target =[0.05, 0.11966080203486248, -0.3]
            # self.arm_target =[0.05, 0.15072823095761758, -0.3]
            # self.arm_target =[0.05, 0.004928055825915323, -0.3]
            # self.arm_target =[0.05616417952559244, -0.055515207026343605, -0.3]
            # self.arm_target =[0.3, -0.29516530809799624, -0.3]
            # arm_action0=self.arm_chain.inverse_kinematics(self.arm_target ,tori, orientation_mode="Z")
           
            arm_action=self.arm_chain.inverse_kinematics(self.arm_target ,target_orientation, initial_position=self.ik_init, orientation_mode="all")[1:-1]
            self.arm_setup_pos = arm_action
            print(self.arm_setup_pos)
            joint = [0]*8
            for i in range(6):
                joint[i+1]=arm_action[i]
            realpos=self.arm_chain.forward_kinematics(joint)
            print(realpos[:3,3])
            print(self.sim.xpos("ee")[:3])
            
        elif self.IK :#wqwqwq
            if self.IKcnt%10 == 0:
                self.arm_target =[0.3, 0.1, -0.05 ]
                # self.arm_target[0] =  np.clip(self.target_pos[2] + 0.143 - self.sim.qpos()[2] - 0.2,0.05,0.3)#height 0.143 is half the knob plate height
                self.arm_target[0] =  np.clip(0.8*self.sim.xpos("ee")[2] + 0.2*self.target_pos[2]  - self.sim.qpos()[2] - 0.2,0.05,0.3)
                self.arm_target[2] = -0.3 #np.clip(-(self.target_pos[0]-self.sim.qpos()[0] - 0.12),-0.3,0.3)
                self.arm_target[1] = 0#np.clip(self.target_pos[1]-self.sim.qpos()[1], -0.3, 0.3)
                target_orientation = np.eye(3)
                print("target:",self.arm_target)
                tori=[0,0,1]
                arm_action=self.arm_chain.inverse_kinematics(self.arm_target ,target_orientation, initial_position=self.ik_init, orientation_mode="all")
                half_action=[(arm_action[x]+self.ik_init[x])/2 for x in range(8)]
                self.ik_init=arm_action
                self.arm_setup_pos = half_action[1:-1]
                print("joint:",self.arm_setup_pos)
                # self.arm_target[0] = 0.05
                # self.arm_target[2] = -(self.target_pos[0]-self.sim.qpos()[0] - 0.12)
                # self.arm_target[1] = (self.target_pos[1]-self.sim.qpos()[1])
                # # self.arm_target = [0.05,0.14,-0.3]   [ 0.08282294  0.00134902 -0.24482408]
                # # self.arm_target[1]=-0.5*self.arm_init_pos[1]
                # tori=[-1,0,0]
                # tori=[0,0,1]
            self.IKcnt+=1
        # elif self.chickenhead:
            
        else:
            # self.arm_setup_pos = [0,2.0944,-2.0944,0,2*0.5236,0]#[0,0.5236,-2.0944,0,0.5236,0]
            self.arm_setup_pos = self.arm_zero

        ry = self.sim.xpos("robotiq_2f_85_right_follower")[1]
        rx = self.sim.xpos("robotiq_2f_85_right_follower")[0]
        ly = self.sim.xpos("robotiq_2f_85_left_follower")[1]
        if rx > 5.9 and ry < 0.18 and ly > 0.18:
            self.sim.set_gripper(True)
    
            # realpos=self.arm_chain.forward_kinematics(self.arm_setup_pos)
            # target=realpos[:3,3]
            # print('T:',target)
            # arm_action=self.arm_chain.inverse_kinematics(target )[1:-1]
            # print("fuck ac:", arm_action)

    
    
    # 只有x方向对准，高度手算
    def _arm_control(self):
        # print("height ", self.sim.xpos("ee")[2])
        maxheight = 2.3
        diff_h=self.arm_init_pos[2]-self.sim.qpos()[2]-0.2#0.15
        # diff_h=self.arm_height-self.sim.qpos()[2]
        
        # ref_pos, ref_vel = self.get_kin_state()
        # self.arm_target[0]=ref_pos[0] + self.arm_init_pos[0]-self.sim.qpos()[0]

        roll, pitch, yaw = quaternion2euler(self.sim.qpos()[3:7])
        dist = np.sqrt(diff_h**2+self.arm_init_pos[0]**2)
        angle1 = np.arcsin(diff_h/dist)
        target_h = dist*np.sin(angle1+pitch)-0.03
        self.arm_target[2]=target_h#+0.47 stable
        # self.arm_target[2]=0.47
        self.arm_target[0]=0.27681454
        # self.arm_target[0]=0.27#0.58

        # self.arm_target[1]=-0.5*self.arm_init_pos[1]
        tori=[-1,0,0]

        arm_action=self.arm_chain.inverse_kinematics(self.arm_target ,tori, orientation_mode="Z")[1:-1]
        # print(arm_action)

        arm_action[0]=0
        arm_action[3]=0
        arm_action[5]=0
        self.arm_setup_pos = arm_action#self.arm_zero


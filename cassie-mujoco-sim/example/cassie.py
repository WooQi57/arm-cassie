# isolated cassie env
from cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
from quaternion_function import *
from phase_function import *

from math import floor
import gym
from gym import spaces
import numpy as np
import os
import random
import copy

import pickle

import torch

class CassieEnv(gym.Env):
    def __init__(self, simrate=60, dynamics_randomization=True,
                 learn_gains=False, visual=True,
                 config="../model/cassie.xml", history=0, **kwargs):
        super(CassieEnv, self).__init__()
        self.config = config
        self.visual = visual
        self.sim = CassieSim(self.config)
        if self.visual:
            self.vis = CassieVis(self.sim)
        self.dynamics_randomization = dynamics_randomization

        # Observation space and State space
        self.observation_space, self.clock_inds, self.mirrored_obs = self.set_up_state_space()
        self.action_space = spaces.Box(low=np.array([-3.14]*10), high=np.array([3.14]*10))
        

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])
        self.u = pd_in_t()

        self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]


        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()
        self.simrate = simrate  # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.time    = 0        # number of time steps in current episode
        self.phase   = 0        # portion of the phase the robot is in
        self.counter = 0        # number of phase cycles completed in episode

        self.strict_relaxer = 0.1
        self.early_reward = False

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        # CONFIGURE OFFSET for No Delta Policies
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        
        self.max_speed = 4.0
        self.min_speed = -0.3
        self.max_side_speed  = 0.3
        self.min_side_speed  = -0.3

        # global flat foot orientation, can be useful part of reward function:
        self.neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
        
        # various tracking variables for reward funcs
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient = 0
        self.r_foot_orient = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0

        # TODO: should this be mujoco tracking var or use state estimator. real command interface will use state est
        # Track pelvis position as baseline for pelvis tracking command inputs
        self.last_pelvis_pos = self.sim.qpos()[0:3]

        #### Dynamics Randomization ####
        
        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03
        
        self.encoder_noise = 0.01
        
        self.damping_low = 0.3
        self.damping_high = 5.0

        self.mass_low = 0.5
        self.mass_high = 1.5

        self.fric_low = 0.4
        self.fric_high = 1.1

        self.speed = 4.0
        self.side_speed = 0.0
        self.orient_add = 0

        # Record default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass = self.sim.get_body_mass()
        self.default_ipos = self.sim.get_body_ipos()
        self.default_fric = self.sim.get_geom_friction()
        self.default_rgba = self.sim.get_geom_rgba()
        self.default_quat = self.sim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        # Keep track of actions, torques

        # self.prev_action = None
        # self.curr_action = None
        # self.prev_torque = None



    def set_up_state_space(self):

        full_state_est_size = 40
        speed_size     = 2      # x speed, y speed
        clock_size     = 2      # sin, cos
        
        
        base_mir_obs = np.array([0.1, 1, -2, 3, -4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, -16, 17, -18, 19, -20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, -32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42])
        obs_size = full_state_est_size
                
        append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size+speed_size)])
        mirrored_obs = np.concatenate([base_mir_obs, append_obs])
        clock_inds = append_obs[0:clock_size].tolist()
        obs_size += clock_size + speed_size
        
        observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(full_state_est_size,))
        mirrored_obs = mirrored_obs.tolist()

        return observation_space, clock_inds, mirrored_obs
        
    def rotate_to_orient(self, vec):
        quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def step_simulation(self,action):
        target = action + self.offset
        target -= self.motor_encoder_noise
        self.u = pd_in_t()

        # foot_pos = np.zeros(6)
        # self.sim.foot_pos(foot_pos)
        # prev_foot = copy.deepcopy(foot_pos)
        
        for i in range(5):
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)
            # self.sim.foot_pos(foot_pos)
            # self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
            # self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
            

    def step(self, action):
        
        # self.l_foot_frc = 0
        # self.r_foot_frc = 0
        # foot_pos = np.zeros(6)
        # self.l_foot_pos = np.zeros(3)
        # self.r_foot_pos = np.zeros(3)
        # self.l_foot_orient_cost = 0
        # self.r_foot_orient_cost = 0
        # self.hiproll_cost = 0
        # self.hiproll_act = 0

        for _ in range(self.simrate):            
            self.step_simulation(action)

            # # Foot Force Tracking
            # foot_forces = self.sim.get_foot_forces()
            # self.l_foot_frc += foot_forces[0]
            # self.r_foot_frc += foot_forces[1]
            # # Relative Foot Position tracking
            # self.sim.foot_pos(foot_pos)
            # self.l_foot_pos += foot_pos[0:3]
            # self.r_foot_pos += foot_pos[3:6]
            # # Foot Orientation Cost
            # self.l_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.sim.xquat("left-foot")) ** 2)
            # self.r_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.sim.xquat("right-foot")) ** 2)
            # # Hip Yaw velocity cost
            # self.hiproll_cost += (np.abs(self.qvel[6]) + np.abs(self.qvel[19])) / 3
            # if self.prev_action is not None:
            #     self.hiproll_act += 2*np.linalg.norm(self.prev_action[[0, 5]] - action[[0, 5]])
            # else:
            #     self.hiproll_act += 0

        # self.l_foot_frc              /= self.simrate
        # self.r_foot_frc              /= self.simrate
        # self.l_foot_pos              /= self.simrate
        # self.r_foot_pos              /= self.simrate
        # self.l_foot_orient_cost      /= self.simrate
        # self.r_foot_orient_cost      /= self.simrate
        # self.hiproll_cost            /= self.simrate
        # self.hiproll_act             /= self.simrate

        obs = self.get_state()
        height = self.qpos[2]
        self.curr_action = action
        self.time  += 1

        if height < 0.4 or height > 3.0:
            done = True
        else:
            done = False

        # if self.prev_action is None:
        #     self.prev_action = action
        # if self.prev_torque is None:
        #     self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])
        # self.prev_action = action
        # self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])
        if self.visual:
            self._render()
        reward = self.compute_reward(action)
        # if reward < 0.3:
        #     done = True

        return obs, reward, done, {}

    def reset(self):
        print('----reset----')
        self.speed = 4.0 # np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = 0.0 # np.random.uniform(self.min_side_speed, self.max_side_speed)

        self.time = 0
        self.counter = 0

        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping

            pelvis_damp_range = [[damp[0], damp[0]],
                                [damp[1], damp[1]],
                                [damp[2], damp[2]],
                                [damp[3], damp[3]],
                                [damp[4], damp[4]],
                                [damp[5], damp[5]]]  # 0->5

            hip_damp_range = [[damp[6]*self.damping_low, damp[6]*self.damping_high],
                              [damp[7]*self.damping_low, damp[7]*self.damping_high],
                              [damp[8]*self.damping_low, damp[8]*self.damping_high]]          # 6->8 and 19->21

            achilles_damp_range = [[damp[9]*self.damping_low, damp[9]*self.damping_high],
                                   [damp[10]*self.damping_low, damp[10]*self.damping_high],
                                   [damp[11]*self.damping_low, damp[11]*self.damping_high]]   # 9->11 and 22->24

            knee_damp_range     = [[damp[12]*self.damping_low, damp[12]*self.damping_high]]   # 12 and 25
            shin_damp_range     = [[damp[13]*self.damping_low, damp[13]*self.damping_high]]   # 13 and 26
            tarsus_damp_range   = [[damp[14]*self.damping_low, damp[14]*self.damping_high]]   # 14 and 27

            heel_damp_range     = [[damp[15], damp[15]]]                                      # 15 and 28
            fcrank_damp_range   = [[damp[16]*self.damping_low, damp[16]*self.damping_high]]   # 16 and 29
            prod_damp_range     = [[damp[17], damp[17]]]                                      # 17 and 30
            foot_damp_range     = [[damp[18]*self.damping_low, damp[18]*self.damping_high]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass
            pelvis_mass_range      = [[self.mass_low*m[1], self.mass_high*m[1]]]   # 1
            hip_mass_range         = [[self.mass_low*m[2], self.mass_high*m[2]],   # 2->4 and 14->16
                                    [self.mass_low*m[3], self.mass_high*m[3]],
                                    [self.mass_low*m[4], self.mass_high*m[4]]]

            achilles_mass_range    = [[self.mass_low*m[5], self.mass_high*m[5]]]    # 5 and 17
            knee_mass_range        = [[self.mass_low*m[6], self.mass_high*m[6]]]    # 6 and 18
            knee_spring_mass_range = [[self.mass_low*m[7], self.mass_high*m[7]]]    # 7 and 19
            shin_mass_range        = [[self.mass_low*m[8], self.mass_high*m[8]]]    # 8 and 20
            tarsus_mass_range      = [[self.mass_low*m[9], self.mass_high*m[9]]]    # 9 and 21
            heel_spring_mass_range = [[self.mass_low*m[10], self.mass_high*m[10]]]  # 10 and 22
            fcrank_mass_range      = [[self.mass_low*m[11], self.mass_high*m[11]]]  # 11 and 23
            prod_mass_range        = [[self.mass_low*m[12], self.mass_high*m[12]]]  # 12 and 24
            foot_mass_range        = [[self.mass_low*m[13], self.mass_high*m[13]]]  # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.0
            com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

            fric_noise = []
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for _ in range(int(len(self.default_fric)/3)):
                fric_noise += [translational, torsional, rolling]

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
        else:
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_geom_friction(self.default_fric)

        if self.dynamics_randomization:
            geom_plane = [np.random.uniform(-self.max_roll_incline, self.max_roll_incline), np.random.uniform(-self.max_pitch_incline, self.max_pitch_incline), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            geom_quat  = list(quat_plane) + list(self.default_quat[4:])
            self.sim.set_geom_quat(geom_quat)
        else:
            self.sim.set_geom_quat(self.default_quat)
    
        # # reset mujoco tracking variables
        # self.l_foot_frc = 0
        # self.r_foot_frc = 0
        # self.l_foot_orient_cost = 0
        # self.r_foot_orient_cost = 0
        # self.hiproll_cost = 0
        # self.hiproll_act = 0
        self.sim.set_const()
        return self.get_state()

    def get_state(self):
        self.qpos = np.copy(self.sim.qpos())  # dim=35 see cassiemujoco.h for details
        self.qvel = np.copy(self.sim.qvel())  # dim=32
        # print('qpos:',self.qpos)
        # print('qvel:',self.qvel)

        '''
		Position [1], [2] 				-> Pelvis y, z
				 [3], [4], [5], [6] 	-> Pelvis Orientation qw, qx, qy, qz
				 [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [14]					-> Left Knee   	(Motor[3])
				 [15]					-> Left Shin   	(Joint[0])
				 [16]					-> Left Tarsus 	(Joint[1])
				 [20]					-> Left Foot   	(Motor[4], Joint[2])
				 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [28]					-> Rigt Knee   	(Motor[8])
				 [29]					-> Rigt Shin   	(Joint[3])
				 [30]					-> Rigt Tarsus 	(Joint[4])
				 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        '''
		Velocity [0], [1], [2] 			-> Pelvis x, y, z
				 [3], [4], [5]		 	-> Pelvis Orientation wx, wy, wz
				 [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [12]					-> Left Knee   	(Motor[3])
				 [13]					-> Left Shin   	(Joint[0])
				 [14]					-> Left Tarsus 	(Joint[1])
				 [18]					-> Left Foot   	(Motor[4], Joint[2])
				 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [25]					-> Rigt Knee   	(Motor[8])
				 [26]					-> Rigt Shin   	(Joint[3])
				 [27]					-> Rigt Tarsus 	(Joint[4])
				 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
		'''
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        return np.concatenate([self.qpos[pos_index], self.qvel[vel_index]])

        # Update orientation
        # new_orient = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
        # new_translationalVelocity = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
        # new_translationalAcceleleration = self.rotate_to_orient(self.cassie_state.pelvis.translationalAcceleration[:])
        
        # motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
        # joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise
        # print("item 1 pelvis height:", self.cassie_state.pelvis.position[0],self.cassie_state.pelvis.position[1],self.cassie_state.pelvis.position[2])
        # print("item 2 pelvis orientation:", new_orient)
        # print("item 3 actuated joint positions:", motor_pos)
        # print("item 4 pelvis translational velocity:", new_translationalVelocity)
        # print("item 5 pelvis rotational velocity:", self.cassie_state.pelvis.rotationalVelocity[:])
        # print("item 6 actuated joint velocities:", self.cassie_state.motor.velocity[:])
        # print("item 7 pelvis translational acceleration:", new_translationalAcceleleration)
        # print("item 8 unactuated joint positions:", joint_pos)
        # print("item 9 unactuated joint velocities:", self.cassie_state.joint.velocity[:])
        # print("height of terrain:",self.cassie_state.terrain.height)
        # robot_state = np.concatenate([
        #     [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],  # pelvis height         0.8
        #     new_orient,                                         # pelvis orientation                            in quaternion
        #     motor_pos,                                          # actuated joint positions                      dim=10
        #     new_translationalVelocity,                          # pelvis translational velocity                 dim=3
        #     self.cassie_state.pelvis.rotationalVelocity[:],     # pelvis rotational velocity                    dim=3
        #     self.cassie_state.motor.velocity[:],                # actuated joint velocities                     dim=10
        #     new_translationalAcceleleration,                    # pelvis translational acceleration             dim=3
        #     joint_pos,                                          # unactuated joint positions                    dim=6
        #     self.cassie_state.joint.velocity[:]                 # unactuated joint velocities                   dim=6
        # ])
        
        # self.observation_space = np.concatenate([robot_state, [self.speed, self.side_speed]])
        # return self.observation_space

    def compute_reward(self, action):
        joint_penalty = np.sum(action * action)
        orientation_penalty = (self.qpos[4])**2+(self.qpos[5])**2+(self.qpos[6])**2
        vel_penalty = (self.speed - self.qvel[0])**2 + (self.side_speed - self.qvel[1])**2 + (self.qvel[2])**2
        spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
        spring_penalty *= 1000
        reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-vel_penalty)+0.1*np.exp(-orientation_penalty)+0.1*np.exp(-spring_penalty)

        return reward

    def _render(self):        
        return self.vis.draw(self.sim)

    def render(self):        
        return 

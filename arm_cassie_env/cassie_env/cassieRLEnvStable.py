from numpy import dtype
from .cassieRLEnvMirrorIKTraj import *
from.cassieRLEnvMirror import *

class cassieRLEnvStable(cassieRLEnvMirror):
    def __init__(self,visual=False,setforce=-1,fix=True):
        super().__init__(visual=visual)
        self.action_space = spaces.Box(low=np.float32(np.array([-3.14/3]*10)), high=np.float32(np.array([3.14/3]*10)))
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(85+6,))

        self.prev_vel=[]
        self.cur_vel=[]
        self.prev_joint=[]
        self.acc=0
        self.rew_acc_buf=0
        self.rew_acc=0

        self.prev_body_vel=None
        self.cur_body_vel=[]
        self.body_acc=0
        self.rew_body_acc_buf=0
        self.rew_body_acc=0
        self.prev_action = []
        self.rew_action = 0
        self.rew_action_buf = 0

        self.acbuf = True
        self.noisy=True
        self.delay=True
        self.buffer_size = 20

        self.hiproll_cost = 0
        self.termination=False

        self.setforce=setforce
        self.arm_zero=np.array([0,-0.5236,-2.0944,0,0,0])  # reset arm wqwqwq
        self.arm_zero=np.array([0,2.0944,-2.0944,0,2*0.5236,0])
        self.arm_zero=np.array([0,1.57-0.51,-1.57-0.51,0,0,0]) 
        # self.arm_zero=np.array([0,-1.4,1.4,0,0,0])  # light

        self.fix = fix  # fix arm or not
        self.isplay=False  # survive longer
        self.test = False  # hang the robot

    def step_simulation(self, action):


        #self.sim.perturb_mass()
        #self.sim.get_mass()
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())
        # self.height_rec.append(qpos[2])

        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.cur_vel=qvel[vel_index]
        self.cur_body_vel=qvel[0]

        if len(self.prev_vel)==10:
            self.acc+=np.sum(np.square(self.cur_vel-self.prev_vel))
        self.prev_vel = np.copy(self.cur_vel)

        if self.prev_body_vel is not None :
            body_dv=(self.cur_body_vel-self.prev_body_vel)**2
            self.body_acc+=body_dv
        self.prev_body_vel = np.copy(self.cur_body_vel)

        ref_pos, ref_vel = self.get_kin_next_state()
        if self.phase < 14:
            target = action[0:10] + ref_pos[pos_index]
        else:
            mirror_action = np.zeros(10)
            mirror_action[0:5] = np.copy(action[5:10])
            mirror_action[5:10] = np.copy(action[0:5])
            mirror_action[0] = -mirror_action[0]
            mirror_action[1] = -mirror_action[1]
            mirror_action[5] = -mirror_action[5]
            mirror_action[6] = -mirror_action[6]
            target = mirror_action + ref_pos[pos_index]
        if self.test:
            self.sim.set_upright_test()
        self.u = pd_in_t()
        for i in range(5):
            self.u.leftLeg.motorPd.torque[i] = 0 # Feedforward torque
            self.u.leftLeg.motorPd.pTarget[i] = target[i]
            self.u.leftLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dTarget[i] = 0
            self.u.leftLeg.motorPd.dGain[i] = self.D[i]
            self.u.rightLeg.motorPd.torque[i] = 0 # Feedforward torque
            self.u.rightLeg.motorPd.pTarget[i] = target[i+5]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i+5]
            self.u.rightLeg.motorPd.dTarget[i] = 0
            self.u.rightLeg.motorPd.dGain[i] = self.D[i+5]

        # ApGain = [200, 150,150,50,10,1] 
        # AdGain = [20,20,15,0.5,0.5,0.1]
        
        # ApGain = [0] *6
        # AdGain = [0]*6
        # self.arm_setup_pos = [0,-1.047,-2.618,0,0,0]
        # self.arm_setup_pos = [0,0,-1.57,0,0,0]

        # uncomment to train with arm wqwqwq
        if self.fix == True:
            ApGain = [3000, 5000,5000,750,5000,100] 
            AdGain = [20,200,30,2,1,0]

            # light
            # ApGain = [100,100,100,10,10,30]
            # AdGain = [20,20,20,1,6,0]
            self.arm_setup_pos = self.arm_zero
            self.sim.sim_arm(self.arm_setup_pos, ApGain, AdGain)

        self.state_buffer.append(self.sim.step_pd(self.u))
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)
        
        self.cassie_state = self.state_buffer[len(self.state_buffer) - 1]

    def step(self, action):
        #wqwqwq
        if self.acbuf:
            self.action_buf.append(np.copy(action))
            if len(self.action_buf)>3:				
                self.action_buf.pop(0)		
                # action = np.mean(self.action_buf,axis=0)
                action = self.action_buf[0]*0.1 + self.action_buf[1]*0.2+self.action_buf[2]*0.7

        self.hiproll_cost = 0
        
        for _ in range(self.control_rate):
            self.step_simulation(action)
            qvel = np.copy(self.sim.qvel())
            self.hiproll_cost += (np.abs(qvel[6]) + np.abs(qvel[13])) / 3
        self.hiproll_cost            /= self.control_rate
        # print("ee:",self.sim.xpos("ee"))
        # print("base:",self.sim.xpos("base_link"))
        # print("pelvis",self.sim.xpos("cassie-pelvis"))
        height = self.sim.qpos()[2]
        pos = self.sim.qpos()[0:2]
        comPos = np.linalg.norm(pos)
        self.time += 1
        self.phase += 1

        if self.phase >= self.max_phase:
            self.phase = 0
            self.counter +=1
        # print("height", height)

        if self.speed != 0:
            isstill = True
        else:
            isstill = comPos < 2 # pos[0] < 1 and pos[0] > -1
        # wqwqwq
        self.termination = not(height > 0.4 and height < 100.0 and isstill)
        if self.isplay:
            self.termination = not(height > 0.4 and height < 100.0 )

        done = self.termination or self.time >= self.time_limit
        if self.isplay:
            self.termination = not(height > 0.4 and height < 100.0 )
            done = self.termination
        yaw = quat2yaw(self.sim.qpos()[3:7])
        if self.visual:
            self.render()
        reward = self.compute_reward(action)
        #print(reward)
        # if reward < 0.3:
        #     done = True

        self.prev_action=action
        qpos = np.copy(self.sim.qpos())
        joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.prev_joint=qpos[joint_index]
        return self.get_state(), reward, done, {}

    def reset(self):
        if self.time != 0 :
            self.rew_ref_buf = self.rew_ref / self.time
            self.rew_spring_buf = self.rew_spring / self.time
            self.rew_ori_buf = self.rew_ori / self.time
            self.rew_vel_buf = self.rew_vel / self.time
            self.reward_buf = self.reward # / self.time
            self.time_buf = self.time
            self.rew_acc_buf = self.rew_acc / self.time
            self.rew_body_acc_buf = self.rew_body_acc/self.time
            self.rew_action_buf = self.rew_action / self.time
        
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.rew_cur = 0
        self.reward = 0
        self.omega = 0
        self.acc = 0
        self.rew_acc = 0
        self.body_acc= 0
        self.rew_body_acc = 0
        self.rew_action = 0

        self.height_rec=[]
    
        self.orientation = 0
        if not self.isplay:
            # self.speed = (random.randint(1, 10)) / 10.0
            # self.speed = random.choice([0,1])  #
            self.speed = 1  #wqwqwq
            self.side_speed = 0.0

        # arm_choice = random.choice([0,1])
        # if arm_choice:  #wqwqwq
        #     self.arm_zero=np.array([0,2.0944,-2.0944,0,2*0.5236,0])
        # else:
        #     self.arm_zero=np.array([0,1.57-0.51,-1.57-0.51,0,0,0]) 

        # orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
        orientation = 0
        quaternion = euler2quat(z=orientation, y=0, x=0)
        # self.phase = random.randint(0, 27)
        self.phase = 0 #wqwqwq
        self.time = 0
        self.counter = 0
        cassie_sim_free(self.sim.c)
        self.sim.c = cassie_sim_init(self.model.encode('utf-8'), False)

        # set force wqwqwq
        self.torque = [random.uniform(-0,0),random.uniform(-0,0),random.uniform(-0,0)]  #y-5 5
        self.force = [random.uniform(-0,0)]*3
        totalforce = self.force+self.torque
        # print(totalforce)
        if self.setforce != -1:
            totalforce = self.setforce
        
        chs=np.array([
            [-1.40375984,  1.85737509, -1.5717031 ,  3.14      , -0.28749207,   -1.40222759],
            [ 1.50926031,  1.85850805, -1.5517621 , -0.05938497,  0.26744237,   -1.57      ],
            [ 0.79022005,  1.81113796, -1.3215943 , -0.78135907, -0.00499692,   -1.57      ],
            [-0.78381726,  1.82780641, -1.22545797,  3.14      ,  0.08832286,  -0.78222309],
            [0,2.0944,-2.0944,0,2*0.5236,0],
            [0,1.57-0.51,-1.57-0.51,0,0,0]
        ])
        # self.arm_setup_pos = random.choice(chs)



        if self.visual:
            print(self.speed)
            print("set wrench:",totalforce)
            # print("upper bound:", self.force_ub)
            # print("reward_buf:",self.reward_buf)
            
        self.sim.apply_force(totalforce)

        qpos, qvel = self.get_kin_state()
        # qvel[0] = (random.randint(1, 7)) / 10.0
        # qvel[1] = (random.randint(-3,3)) / 10.0
        qvel[0]=0.6527778506278992  #wqwqwq
        # qvel[0]=0.8
        if self.visual:
            print("v0:   x:",qvel[0],"y:",qvel[1])

        qpos[3:7] = quaternion
        # self.arm_zero=np.array([0,-0.5236,-2.0944,0,0,0])  # reset arm
        # self.arm_zero=np.array([0,-1.047,-2.618,0,0,0])  # reset arm
        
        # self.arm_zero=np.array([0,0,-1.57,0,0,0])
        qpos=np.concatenate([qpos,self.arm_zero],axis=0)
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        self.cassie_state = self.sim.step_pd(self.u)
        self.hiproll_cost = 0
        return self.get_state()

    def get_state(self):
        if len(self.state_buffer) > 0:
            random_index = random.randint(0, len(self.state_buffer)-1)
            state = self.state_buffer[random_index]
        else:
            state = self.cassie_state
        rp ,rv = self.get_kin_next_state()
        ref_pos= np.copy(rp)
        ref_vel=np.copy(rv) 
        wrench_sensor = np.array( [x  for x in self.sim.arm_wrench()])
        if self.phase < 14:
            pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
            vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
            quaternion = euler2quat(z=self.orientation, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
            #print(new_orientation)
            new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
            #print(new_translationalVelocity)
            new_translationalAcceleration = rotate_by_quaternion(state.pelvis.translationalAcceleration[:], iquaternion)
            new_rotationalVelocity = rotate_by_quaternion(state.pelvis.rotationalVelocity[:], quaternion)
            useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], new_translationalAcceleration[:], state.joint.position[:], state.joint.velocity[:]]))

            return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index],  wrench_sensor])
        else:
            pos_index = np.array([2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
            vel_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
            ref_vel[1] = -ref_vel[1]
            euler = quaternion2euler(ref_pos[3:7])
            euler[0] = -euler[0]
            euler[2] = -euler[2]
            ref_pos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])
            quaternion = euler2quat(z=-self.orientation, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)

            pelvis_euler = quaternion2euler(np.copy(state.pelvis.orientation[:]))
            pelvis_euler[0] = -pelvis_euler[0]
            pelvis_euler[2] = -pelvis_euler[2]
            pelvis_orientation = euler2quat(z=pelvis_euler[2],y=pelvis_euler[1],x=pelvis_euler[0])

            translational_velocity = np.copy(state.pelvis.translationalVelocity[:])
            translational_velocity[1] = -translational_velocity[1]

            translational_acceleration = np.copy(state.pelvis.translationalAcceleration[:])
            translational_acceleration[1] = -translational_acceleration[1]

            rotational_velocity = np.copy(state.pelvis.rotationalVelocity)
            rotational_velocity[0] = -rotational_velocity[0]
            rotational_velocity[2] = -rotational_velocity[2]

            motor_position = np.zeros(10)
            motor_position[0:5] = np.copy(state.motor.position[5:10])
            motor_position[5:10] = np.copy(state.motor.position[0:5])
            motor_position[0] = -motor_position[0]
            motor_position[1] = -motor_position[1]
            motor_position[5] = -motor_position[5]
            motor_position[6] = -motor_position[6]

            motor_velocity = np.zeros(10)
            motor_velocity[0:5] = np.copy(state.motor.velocity[5:10])
            motor_velocity[5:10] = np.copy(state.motor.velocity[0:5])
            motor_velocity[0] = -motor_velocity[0]
            motor_velocity[1] = -motor_velocity[1]
            motor_velocity[5] = -motor_velocity[5]
            motor_velocity[6] = -motor_velocity[6]

            joint_position = np.zeros(6)
            joint_position[0:3] = np.copy(state.joint.position[3:6])
            joint_position[3:6] = np.copy(state.joint.position[0:3])

            joint_velocity = np.zeros(6)
            joint_velocity[0:3] = np.copy(state.joint.velocity[3:6])
            joint_velocity[3:6] = np.copy(state.joint.velocity[0:3])

            left_toeForce = np.copy(state.rightFoot.toeForce[:])
            left_toeForce[1] = -left_toeForce[1]
            left_heelForce = np.copy(state.rightFoot.heelForce[:])
            left_heelForce[1] = -left_heelForce[1]

            right_toeForce = np.copy(state.leftFoot.toeForce[:])
            right_toeForce[1] = -right_toeForce[1]
            right_heelForce = np.copy(state.leftFoot.heelForce[:])
            right_heelForce[1] = -right_heelForce[1]
            
            new_orientation = quaternion_product(iquaternion, pelvis_orientation)
            new_translationalVelocity = rotate_by_quaternion(translational_velocity, iquaternion)
            new_translationalAcceleration = rotate_by_quaternion(translational_acceleration, iquaternion)
            new_rotationalVelocity = rotate_by_quaternion(rotational_velocity, quaternion)

            useful_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], motor_position, new_translationalVelocity[:], rotational_velocity, motor_velocity, new_translationalAcceleration[:], joint_position, joint_velocity]))
        

            return np.concatenate([useful_state, ref_pos[pos_index], ref_vel[vel_index], wrench_sensor])

    def compute_reward(self,action):
        ref_pos, ref_vel = self.get_kin_state()
        # ref_pos[0]=0#wqwqwq
        # ref_pos[1]=0
        # ref_vel[0]=0
        # ref_vel[1]=0
        weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]
        joint_penalty = 0
        joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        if len(self.prev_action) > 1:
            hip_action_penalty = np.linalg.norm(action[[0,5]]-self.prev_action[[0,5]])
        else:
            hip_action_penalty = 0
        self.prev_action = action
        action_penalty = np.linalg.norm(action)

        for i in range(10):
            error = weight[i] * (ref_pos[joint_index[i]]-self.sim.qpos()[joint_index[i]])**2
            joint_penalty += error*30

        pelvis_pos = np.copy(self.cassie_state.pelvis.position[:])
        com_penalty = 1.2*(pelvis_pos[0]-ref_pos[0])**2 + (pelvis_pos[1]-ref_pos[1])**2 + (self.sim.qvel()[2])**2

        roll, pitch, yaw = quaternion2euler(self.sim.qpos()[3:7])

        orientation_penalty = roll**2+pitch**2+(yaw - self.orientation)**2  #wqwqwq

        spring_penalty = (self.sim.qpos()[15])**2+(self.sim.qpos()[29])**2
        spring_penalty *= 1000

        speed_penalty = (self.sim.qvel()[0] - ref_vel[0])**2 + (self.sim.qvel()[1] - ref_vel[1])**2

        # total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-10*orientation_penalty)-0.00001*self.acc/self.time+0.1*np.exp(-spring_penalty)
        # total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-10*orientation_penalty)-0.000015*self.acc/self.time+0.1*np.exp(-spring_penalty)
        # total_reward = 0.5*np.exp(-joint_penalty)+0.3*np.exp(-com_penalty)+0.1*np.exp(-10*orientation_penalty)-0.00005*self.acc/self.time+0.1*np.exp(-spring_penalty)

        # total_reward = 0.5*np.exp(-joint_penalty)       \
        #                             +0.2*np.exp(-com_penalty)       \
        #                             +0.2*np.exp(-10*orientation_penalty)   \
        #                             +0.05*np.exp(-hip_action_penalty)  \
        #                             -0.0001*self.acc/self.time    \
        #                             +0.1*np.exp(-self.hiproll_cost)  \
        #                             +0.1*np.exp(-spring_penalty)  \

        total_reward = 0.3*np.exp(-joint_penalty)       \
                            +0.2*np.exp(-com_penalty)       \
                            +0.2*np.exp(-10*orientation_penalty)   \
                            +0.05*np.exp(-hip_action_penalty)  \
                            +0.1*np.exp(-self.hiproll_cost)  \
                            +0.1*np.exp(-spring_penalty)  \
                            # -10*self.termination
                            # +0.01*np.exp(-action_penalty)#-0.3*self.body_acc/self.time#wqwqwq

                                    

        self.rew_body_acc+=0.3*self.body_acc/self.time

        self.rew_acc+=0.0001*self.acc/self.time
        self.rew_ref += 0.3*np.exp(-joint_penalty)
        self.rew_spring += 0.1*np.exp(-spring_penalty)
        self.rew_ori += 0.2*np.exp(-10*orientation_penalty)
        self.rew_vel += 0.2*np.exp(-com_penalty)
        self.rew_action += 0.05*np.exp(-hip_action_penalty)

        
        self.reward += total_reward		
        
        return total_reward

    
    def reset_test(self):
        if self.time != 0 :
            self.rew_ref_buf = self.rew_ref / self.time
            self.rew_spring_buf = self.rew_spring / self.time
            self.rew_ori_buf = self.rew_ori / self.time
            self.rew_vel_buf = self.rew_vel / self.time
            self.reward_buf = self.reward # / self.time
            self.time_buf = self.time
            self.rew_acc_buf = self.rew_acc / self.time
            self.rew_body_acc_buf = self.rew_body_acc/self.time
            self.rew_action_buf = self.rew_action / self.time
        
        self.rew_ref = 0
        self.rew_spring = 0
        self.rew_ori = 0
        self.rew_vel = 0
        self.rew_termin = 0
        self.rew_cur = 0
        self.reward = 0
        self.omega = 0
        self.acc = 0
        self.rew_acc = 0
        self.body_acc= 0
        self.rew_body_acc = 0
        self.rew_action = 0

        self.height_rec=[]
    
        self.orientation = 0
        # self.speed = (random.randint(1, 10)) / 10.0
        self.speed = 0
        # orientation = self.orientation + random.randint(-20, 20) * np.pi / 100
        orientation = 0
        quaternion = euler2quat(z=orientation, y=0, x=0)
        self.phase = random.randint(0, 27)
        # self.phase = 0
        self.time = 0
        self.counter = 0
        cassie_sim_free(self.sim.c)
        self.sim.c = cassie_sim_init(self.model.encode('utf-8'), False)
        # set force
        self.force = [0,0,0]#0.2*random.randint(0,10)-1+50
        self.torque = [0,0,0]
        if self.setforce != -1:
            self.force = self.setforce
            
        
        if self.visual:
            print("z-axis force:",self.force)
            # print("upper bound:", self.force_ub)
            # print("reward_buf:",self.reward_buf)

        # self.sim.apply_force([0,0, -self.zforce, 0,  0, 0])

        qpos, qvel = self.get_kin_state()
        qvel[0]=0
        qpos[3:7] = quaternion
        qpos[2]=2
        arm_zero=np.array([0,-1.047,-2.618,0,0,0])  # reset arm
        qpos=np.concatenate([qpos,arm_zero,np.zeros(10)],axis=0)
        # print(qpos)
        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)
        self.cassie_state = self.sim.step_pd(self.u)
        self.hiproll_cost = 0
        return self.get_state()

    def step_test(self, action):
        self.hiproll_cost = 0

        for _ in range(self.control_rate):
            self.step_simulation_test(action)
            qvel = np.copy(self.sim.qvel())
            self.hiproll_cost += (np.abs(qvel[6]) + np.abs(qvel[13])) / 3
        self.hiproll_cost            /= self.control_rate
        # print("ee:",self.sim.xpos("ee"))
        # print("brace:",self.sim.xpos("bracelet_link"))
        # print("pelvis",self.sim.xpos("cassie-pelvis"))
        height = self.sim.qpos()[2]
        self.time += 1
        self.phase += 1

        if self.phase >= self.max_phase:
            self.phase = 0
            self.counter +=1

        done = not(height > 0.4 and height < 100.0) or self.time >= self.time_limit
        yaw = quat2yaw(self.sim.qpos()[3:7])
        if self.visual:
            self.render()

        reward = self.compute_reward(action)
        if reward < 0.3:
            done = True

        self.prev_action=action
        qpos = np.copy(self.sim.qpos())
        joint_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.prev_joint=qpos[joint_index]
        return self.get_state(), reward, done, {}

    def step_simulation_test(self, action):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        pos_index = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        vel_index = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]
        self.cur_vel = qvel[vel_index]
        self.cur_body_vel = qvel[0]

        if len(self.prev_vel) == 10:
            self.acc += np.sum(np.square(self.cur_vel - self.prev_vel))
        self.prev_vel = np.copy(self.cur_vel)

        if self.prev_body_vel is not None:
            body_dv = (self.cur_body_vel - self.prev_body_vel) ** 2
            self.body_acc += body_dv
        self.prev_body_vel = np.copy(self.cur_body_vel)

        ref_pos, ref_vel = self.get_kin_next_state()
        offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        # target = action[0:10] + offset
        target = action[0:10] + ref_pos[pos_index]

        self.u = pd_in_t()
        self.sim.set_upright_test()
        for i in range(5):
            self.u.leftLeg.motorPd.torque[i] = 0  # Feedforward torque
            self.u.leftLeg.motorPd.pTarget[i] = target[i]
            self.u.leftLeg.motorPd.pGain[i] = self.P[i]
            self.u.leftLeg.motorPd.dTarget[i] = 0
            self.u.leftLeg.motorPd.dGain[i] = self.D[i]
            self.u.rightLeg.motorPd.torque[i] = 0  # Feedforward torque
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i + 5]
            self.u.rightLeg.motorPd.dTarget[i] = 0
            self.u.rightLeg.motorPd.dGain[i] = self.D[i + 5]

        # ApGain = [100, 100,100,50,10,1] 
        # AdGain = [10,25,20,0.5,0.5,0.1]
        ApGain = [200, 200,150,50,10,1] 
        AdGain = [30,20,15,0.5,0.5,0.1]

        self.sim.sim_arm(action[10:], ApGain, AdGain)

        self.state_buffer.append(self.sim.step_pd(self.u))
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)

        self.cassie_state = self.state_buffer[len(self.state_buffer) - 1]

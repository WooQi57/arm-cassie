import gym
import torch
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from cassie_env.cassieRLEnvMirror import cassieRLEnvMirror
from cassie_env.cassieRLEnvMirrorIKTraj import cassieRLEnvMirrorIKTraj
from cassie_env.cassieRLEnvStable import cassieRLEnvStable
from cassie_env.cassieArm import cassieArm
import numpy as np

def make_env(env_id):
    def _f():
        if env_id == 0:
            env = cassieRLEnvStable(visual=True)
        else:
            env = cassieRLEnvStable(visual=False)
        return env
    return _f

if __name__ == '__main__':
    envs =[make_env(seed) for seed in range(20)]
    envs = SubprocVecEnv(envs)
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    
    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """
        def __init__(self, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)

        def _on_step(self) -> bool:                
            self.logger.record('reward/ref', np.mean(self.training_env.get_attr('rew_ref_buf')))
            self.logger.record('reward/spring', np.mean(self.training_env.get_attr('rew_spring_buf')))
            self.logger.record('reward/orientation', np.mean(self.training_env.get_attr('rew_ori_buf')))
            self.logger.record('reward/velocity', np.mean(self.training_env.get_attr('rew_vel_buf')))
            self.logger.record('reward/steps', np.mean(self.training_env.get_attr('time_buf')))
            self.logger.record('reward/totalreward', np.mean(self.training_env.get_attr('reward_buf')))  
            self.logger.record('reward/acc', np.mean(self.training_env.get_attr('rew_acc_buf')))    
            # self.logger.record('reward/ee', np.mean(self.training_env.get_attr('rew_ee_buf')))    
            self.logger.record('reward/action', np.mean(self.training_env.get_attr('rew_action_buf')))          
            self.logger.record('reward/perf', np.mean(self.training_env.get_attr('rew_ori_buf'))
                                            + np.mean(self.training_env.get_attr('rew_vel_buf')))
            
            if self.n_calls % 51200 == 0:
                print("Saving model")
                self.model.save(f"./model_saved/{t}/ppo_cassie-{self.n_calls}")

            return True

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[dict(pi=[512, 256, 128], vf=[512, 256, 128])])
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                  net_arch=[dict(pi=[256, 256], vf=[256, 256])])
    # model = PPO("MlpPolicy", envs, verbose=1, n_steps=256, policy_kwargs=policy_kwargs,
    #     batch_size=128,tensorboard_log="./log/")  

    # loadmodel = '/home/yons/cassie-stuff/arm-cassie/arm_cassie_env/model_saved/2022-09-29 14:35:29/ppo_cassie-409600.zip'
    loadmodel = '/home/yons/cassie-stuff/arm-cassie/arm_cassie_env/lightmodel/v1'
    model = PPO.load(loadmodel,env=envs)
    model.is_tb_set = False
    
    model.learn(total_timesteps=6e7,n_eval_episodes=10,callback=TensorboardCallback())
    model.save("./wrenchflatmodel/ppo_saved")




    # model = PPO("MlpPolicy", envs, verbose=1, n_steps=5096, policy_kwargs=policy_kwargs,
    #     batch_size=64,n_epochs=3,tensorboard_log="./log/")  
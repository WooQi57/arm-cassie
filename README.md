# arm_cassie_env
Mujoco210 is required for simulation and should be placed in ~/.mujoco . To build the mujoco simulation, `make` in `./cassie-mujoco-sim` folder. Then copy  `libcassiemujoco.so` to arm-cassie/arm_cassie_env/cassie_m

Stable-baselines3 in required for rl algorithm. Create conda env:

```
conda env create -f environment.yml
conda activate lg
```

Run `python train.py` to train the agent defined in cassieRLEnvStable.py.

Run `python play.py` to test the policy.

Run `python arm.py` to run the task.

XML Models for possible use are in `./models`



https://github.com/osudrl/apex

https://github.com/osudrl/cassie-mujoco-sim

https://github.com/ZhaomingXie/cassie_rl_env

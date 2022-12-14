from .cassieRLEnvMirror import *
import pickle

class cassieRLEnvMirrorIKTraj(cassieRLEnvMirror):
	def __init__(self,visual=False):
		super().__init__(visual=visual)
		self.step_in_place_trajectory = CassieTrajectory("trajectory/more-poses-trial.bin")
		for i in range(841):
			self.step_in_place_trajectory.qpos[i][7:21] = np.copy(self.step_in_place_trajectory.qpos[(i+841)][21:35])
			self.step_in_place_trajectory.qpos[i][12] = -self.step_in_place_trajectory.qpos[i][12]
			self.step_in_place_trajectory.qpos[i][21:35] = np.copy(self.step_in_place_trajectory.qpos[(i+841)][7:21])
			self.step_in_place_trajectory.qpos[i][26] = -self.step_in_place_trajectory.qpos[i][26]
		for i in range(1682):
			self.step_in_place_trajectory.qpos[i][2] = 1.05

	def get_kin_state(self):
		if self.speed < 0:
			phase = self.phase
			if phase >= self.max_phase:
				phase = 0
			phase = self.max_phase - phase
			pose = np.copy(self.step_in_place_trajectory.qpos[phase*self.control_rate])
			pose[0] += self.speed * 1.0 * (self.max_phase-phase) / 28
			vel = np.copy(self.step_in_place_trajectory.qvel[phase*self.control_rate])
			pose[0] += (1.0 - self.step_in_place_trajectory.qpos[0, 0])* self.counter * self.speed
			#print(pose[0])
			pose[1] = 0
			vel[0] = 0.8 * self.speed
		else:
			phase = self.phase
			if phase >= self.max_phase:
				phase = 0
			pose = np.copy(self.step_in_place_trajectory.qpos[phase*self.control_rate])
			pose[0] += self.speed * 1.0 * phase / 28
			vel = np.copy(self.step_in_place_trajectory.qvel[phase*self.control_rate])
			pose[0] += (1.0 - self.step_in_place_trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel[0] = 0.8 * self.speed
		#print(pose[0])
		pose[3] = 1
		pose[4:7] = 0
		pose[7] = 0
		pose[8] = 0
		pose[21] = 0
		pose[22] = 0
		vel[6] = 0
		vel[7] = 0
		vel[19] = 0
		vel[20] = 0
		return pose, vel
	def get_kin_next_state(self):
		if self.speed < 0:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			phase = self.max_phase - phase
			pose = np.copy(self.step_in_place_trajectory.qpos[phase*self.control_rate])
			pose[0] += self.speed * 1.0 * (self.max_phase-phase) / 28
			vel = np.copy(self.step_in_place_trajectory.qvel[phase*self.control_rate])
			pose[0] += (1.0- self.step_in_place_trajectory.qpos[0, 0])* self.counter * self.speed
			#print(pose[0])
			pose[1] = 0
			vel[0] = 0.8 * self.speed
		else:
			phase = self.phase + 1
			if phase >= self.max_phase:
				phase = 0
			pose = np.copy(self.step_in_place_trajectory.qpos[phase*self.control_rate])
			pose[0] += self.speed * 1.0 * phase / 28
			vel = np.copy(self.step_in_place_trajectory.qvel[phase*self.control_rate])
			pose[0] += (1.0 - self.step_in_place_trajectory.qpos[0, 0])* self.counter * self.speed
			pose[1] = 0
			vel[0] = 0.8 * self.speed
		#print(pose[0])
		pose[3] = 1
		pose[4:7] = 0
		pose[7] = 0
		pose[8] = 0
		pose[21] = 0
		pose[22] = 0
		vel[6] = 0
		vel[7] = 0
		vel[19] = 0
		vel[20] = 0
		return pose, vel
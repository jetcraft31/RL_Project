import gym


class MountainCarWrapper(gym.Wrapper):
	"""
	Specific wrapper to scale the reward of the MountainCarContinuous environment
	"""
	def __init__(self, env):
		super(MountainCarWrapper, self).__init__(env)

	def step(self, action): #do nothing, not used in simu for now.


		if isinstance(action,list) :
			act = action[0]
		else:
			act = action
			
		next_state, reward, done, y = self.env.step(act)
		#reward = next_state[0] + 0.5
		return next_state, reward, done, y

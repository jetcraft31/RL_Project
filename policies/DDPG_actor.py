import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))
		
		
	@torch.jit.export
	def select_action(self, state: List[float], deterministic: bool=False) -> List[float]:
		"""
		Compute an action or vector of actions given a state or vector of states
		:param state: the input state(s)
		:param deterministic: whether the policy should be considered deterministic or not
		:return: the resulting action(s)
		"""
		state = torch.tensor(state)
		action = self.forward(state)
		#action = np.clip(action,-1,1)
		act: List[float] = action.data.tolist()
		return act

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, l1_units=700, l2_units=400):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, l1_units)
		self.l2 = nn.Linear(l1_units + action_dim, l2_units)
		self.l3 = nn.Linear(l2_units, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

USE_CUDA = torch.cuda.is_available()
device = torch.device("cpu" if USE_CUDA else "cpu")  

# DQN
class Net(nn.Module):
    def __init__(self, action_n, state_n):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_n, 50).to(device)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, action_n).to(device)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

    @torch.jit.export
    def select_action(self, state: List[float], deterministic: bool=False) -> List[int]:

        state = torch.tensor(state)
        action = self.forward(state)

        action = torch.argmax(action).item()
        #action = np.clip(action,-1,1)
        act: List[int] = [int(action)]
        
        return act
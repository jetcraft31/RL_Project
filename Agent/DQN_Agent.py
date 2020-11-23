import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from policies.DQN_policy import Net

USE_CUDA = torch.cuda.is_available()
device = torch.device("cpu" if USE_CUDA else "cpu")        

class DQN(object):
    def __init__(self, action_n, state_n, env_shape, args, learning_rate=0.01, reward_decay=0.9) :
        self.eval_net, self.target_net = Net(action_n=action_n, state_n=state_n), Net(action_n=action_n, state_n=state_n)

        self.action_n = action_n
        self.state_n = state_n
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = args.E_GREEDY
        self.env_shape = env_shape
        self.args = args


        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.args.MEMORY_CAPACITY, self.state_n * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def select_action(self,x):

        x = torch.FloatTensor(x.reshape(1, -1)).to(device)

        #x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            action = self.eval_net.select_action(x)[0]
            #action = action[0] if self.env_shape == 0 else action.reshape(self.env_shape)
        else:  # random
            action = np.random.randint(0, self.action_n)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.args.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.args.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.args.MEMORY_CAPACITY, self.args.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_n]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.state_n:self.state_n+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.state_n+1:self.state_n+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_n:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.args.BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

    def save_model(self):
        traced = torch.jit.script(self.eval_net)
        torch.jit.save(traced, "data/policies/MountainCar-v0#Jérome_DQN#discrete#200#_.zip")
        #torch.save(self.eval_net.state_dict(), 'model/DQN/eval_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))
        #torch.save(self.target_net.state_dict(), 'model/DQN/target_{}_{}_{}.pkl'.format(self.lr, self.epsilon, BATCH_SIZE))

    def load_model(self, model_name):
        self.eval_net.load_state_dict(torch.load(model_name))
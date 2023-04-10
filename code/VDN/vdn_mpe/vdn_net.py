import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
class VDN:
    def __init__(self, agents_num, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                  epsilon, target_update, device):

        # 智能体数量
        self.agents_num = agents_num
        # 状态维度
        self.state_dim = state_dim
        # 隐藏层维度
        self.hidden_dim = hidden_dim
        # 动作维度
        self.action_dim = action_dim
        # 学习率
        self.learning_rate = learning_rate
        # 折扣因子
        self.gamma = gamma
        # epsilon-贪婪策略
        self.epsilon = epsilon
        # 目标网络更新频率
        self.target_update = target_update
        # 设备
        self.device = device
        self.count = 0  # 计数器,记录更新次数
        
        Qnet_input_dim = state_dim * self.agents_num 

        # 神经网络
        # 所有智能体共用一个决策q网络，但是每个智能体有一个独立的target_q网络
        self.eval_qnet = Qnet(Qnet_input_dim, hidden_dim,
                          self.action_dim).to(device)   # 每个agent选动作的网络
        self.target_qnet = []
        for i in range(self.agents_num):
            self.target_qnet.append(Qnet(Qnet_input_dim, hidden_dim,
                          self.action_dim).to(device))  # 每个agent选动作的网络



       
        # 让target_net和eval_net的网络参数相同
        for i in range(self.agents_num):
            self.target_qnet[i].load_state_dict(self.eval_qnet.state_dict())

        self.optimizer = torch.optim.Adam(self.eval_qnet.parameters(), lr=self.learning_rate)

        print('Init alg VDN')

    def update(self, transition_dict):
        """更新Q网络参数"""
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions1 = torch.tensor(transition_dict['actions1']).view(-1, 1).to(
            self.device)
        actions2 = torch.tensor(transition_dict['actions2']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 计算loss
        q_values_1 = self.eval_qnet(states)
        q_values_2 = self.eval_qnet(states)
        # 分别计算每个智能体的target_q值并加起来
        for i in range(self.agents_num):
            next_q_value= self.target_qnet[i](next_states)
            if i == 0:
                next_q_values = next_q_value
            else:
                next_q_values = next_q_values + next_q_value

        with torch.no_grad():
            max_next_q_values = next_q_values.max(dim=1)[0].unsqueeze(1) 
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        # # detach()函数将target_q_values从计算图中分离出来，防止梯度传播
        # target_q_values = target_q_values.detach()
        # 获取对应的q值
        q_values1 = q_values_1.gather(1, actions1)
        q_values2 = q_values_2.gather(1, actions2)
        q_values = q_values1 + q_values2
        loss = nn.functional.mse_loss(q_values, target_q_values)

        # 反向传播更新参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        

        # 更新target网络
        if self.count % self.target_update == 0:
            for i in range(self.agents_num):
                self.target_qnet[i].load_state_dict(self.eval_qnet.state_dict())
        self.count += 1
      


    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)

        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.eval_qnet(state).argmax().item()

        return action
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
print("package loaded")
'''导入环境'''
def make_env(scenario_name):
    # 从环境文件脚本中创建环境
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation)
    return env

env_id = "simple_adversary"
env = make_env(env_id)

state_dims = []
action_dims = []
for action_space in env.action_space:
    action_dims.append(action_space.n)

for state_space in env.observation_space:
    state_dims.append(state_space.shape[0])

print("state_dims:", state_dims)
print("action_dims:", action_dims)

'''定义测试函数'''
def evaluate(env_id, agent_adversary,agent_normal, n_episode=10, episode_length=25):
    # 对学习的策略进行评估,此时不会进行探索
    env = make_env(env_id)
    returns = np.zeros(len(env.agents))
    for _ in range(n_episode):
        state = env.reset()
        for t_i in range(episode_length):
            state_1 = state[0]
            state_2 = state[1]
            state_3 = state[2]
            a_1 = agent_adversary.take_action(state_1)
            a_2 = agent_normal.take_action(state_2)
            a_3 = agent_normal.take_action(state_3)
            a_1_one_hot = np.eye(action_dims[0])[a_1]
            a_2_one_hot = np.eye(action_dims[1])[a_2]
            a_3_one_hot = np.eye(action_dims[1])[a_3]
            actions = [a_1_one_hot, a_2_one_hot, a_3_one_hot]
            state, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

'''定义经验回放池'''

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    

'''定义DQN'''
    
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
            # # 将action转换为独热编码
            # action = np.eye(self.action_dim)[action]
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            # # 将action转换为独热编码
            # action = np.eye(self.action_dim)[action]
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

'''超参数设定'''
num_episodes = 5000
episode_length = 25  # 每条序列的最大长度
hidden_dim = 128
lr = 2e-3
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 1000  # buffer中数据的最小数量,小于该值时不进行训练
batch_size=512
update_interval = 100  # 每隔多少个episode进行一次训练

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
device = torch.device("cpu")
print('device:', device)

replay_buffer_adversary = ReplayBuffer(buffer_size)
replay_buffer_normal_1 = ReplayBuffer(buffer_size)
replay_buffer_normal_2 = ReplayBuffer(buffer_size)



agent_adversary = DQN(state_dims[0], hidden_dim, action_dims[0], lr, gamma, epsilon,
                target_update, device)
#两个智能体共用一个Q网络
agent_normal = DQN(state_dims[1], hidden_dim, action_dims[1], lr, gamma, epsilon,
                target_update, device)

import time

if __name__ == '__main__':
    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()

        for e_i in range(episode_length):
            state_1 = state[0]
            state_2 = state[1]
            state_3 = state[2]
            a_1 = agent_adversary.take_action(state_1)
            a_2 = agent_normal.take_action(state_2)
            a_3 = agent_normal.take_action(state_3)
            a_1_one_hot = np.eye(action_dims[0])[a_1]
            a_2_one_hot = np.eye(action_dims[1])[a_2]
            a_3_one_hot = np.eye(action_dims[1])[a_3]
            actions = [a_1_one_hot, a_2_one_hot, a_3_one_hot]
            next_state, reward, done, _ = env.step(actions)
            # 拆分reward
            r_1 = reward[0]
            r_2 = reward[1]
            r_3 = reward[2]
            # 拆分done
            d_1 = done[0]
            d_2 = done[1]
            d_3 = done[2]
            # 拆分next_state
            next_state_1 = next_state[0]
            next_state_2 = next_state[1]
            next_state_3 = next_state[2]

            # 将transition存入replay buffer
            replay_buffer_adversary.add(state_1, a_1, r_1, next_state_1, d_1)
            replay_buffer_normal_1.add(state_2, a_2, r_2, next_state_2, d_2)
            replay_buffer_normal_2.add(state_3, a_3, r_3, next_state_3, d_3)

            state = next_state

            total_step += 1
            if replay_buffer_adversary.size(
            ) >= minimal_size and total_step % update_interval == 0:
                b_s_1, b_a_1, b_r_1, b_ns_1, b_d_1 = replay_buffer_adversary.sample(batch_size)
                transition_dict_1 = {
                    'states': b_s_1,
                    'actions': b_a_1,
                    'next_states': b_ns_1,
                    'rewards': b_r_1,
                    'dones': b_d_1
                }

                b_s_2, b_a_2, b_r_2, b_ns_2, b_d_2 = replay_buffer_normal_1.sample(batch_size)
                transition_dict_2 = {
                    'states': b_s_2,
                    'actions': b_a_2,
                    'next_states': b_ns_2,
                    'rewards': b_r_2,
                    'dones': b_d_2
                }

                b_s_3, b_a_3, b_r_3, b_ns_3, b_d_3 = replay_buffer_normal_2.sample(batch_size)
                transition_dict_3 = {
                    'states': b_s_3,
                    'actions': b_a_3,
                    'next_states': b_ns_3,
                    'rewards': b_r_3,
                    'dones': b_d_3
                }
                
                agent_adversary.update(transition_dict_1)
                agent_normal.update(transition_dict_2)
                agent_normal.update(transition_dict_3)

        if (i_episode + 1) % 100 == 0:
            ep_returns = evaluate(env_id, agent_adversary,agent_normal, n_episode=100)
            return_list.append(ep_returns)
            print(f"Episode: {i_episode+1}, {ep_returns}")

    '''绘制训练曲线'''
    return_array = np.array(return_list)
    for i, agent_name in enumerate(["adversary_0", "agent_0", "agent_1"]):
        plt.figure()
        plt.plot(
            np.arange(return_array.shape[0]) * 100,
            rl_utils.moving_average(return_array[:, i], 9))
        plt.xlabel("Episodes")
        plt.ylabel("Returns")
        plt.title(f"{agent_name} by IDQN")
        plt.show()
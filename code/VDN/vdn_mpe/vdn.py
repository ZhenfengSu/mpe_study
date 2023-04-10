import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
# 自己定义的库
from vdn_net import VDN
from RepluBuffer import ReplayBuffer,MAReplayBuffer
from dqn_net import DQN
import rl_utils
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
            state_normal = np.concatenate((state_2, state_3), axis=0)
            a_1 = agent_adversary.take_action(state_1)
            a_2 = agent_normal.take_action(state_normal)
            a_3 = agent_normal.take_action(state_normal)
            a_1_one_hot = np.eye(action_dims[0])[a_1]
            a_2_one_hot = np.eye(action_dims[1])[a_2]
            a_3_one_hot = np.eye(action_dims[1])[a_3]
            actions = [a_1_one_hot, a_2_one_hot, a_3_one_hot]
            state, rew, done, info = env.step(actions)
            rew = np.array(rew)
            returns += rew / n_episode
    return returns.tolist()

'''超参数设定'''
num_episodes = 14000
episode_length = 25  # 每条序列的最大长度
hidden_dim = 64
hidden_dim_vdn = 256
lr = 2e-3
lr_vdn = 1e-2
gamma = 0.95
gamma_vdn = 0.95
epsilon = 0.02
epsilon_vdn = 0.02
target_update = 10
buffer_size = 100000
minimal_size = 4000  # buffer中数据的最小数量,小于该值时不进行训练
batch_size=512
batch_size_vdn = 1024
update_interval = 100  # 每隔多少个episode进行一次训练

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
device = torch.device("cpu")
print('device:', device)

replay_buffer_adversary = ReplayBuffer(buffer_size)
replay_buffer_normal = MAReplayBuffer(buffer_size)



agent_adversary = DQN(state_dims[0], hidden_dim, action_dims[0], lr, gamma, epsilon,
                target_update, device)
#两个智能体共用一个Q网络
agent_normal = VDN(2,state_dims[1], hidden_dim_vdn, action_dims[1], lr_vdn , gamma_vdn, epsilon_vdn,target_update, device)


if __name__ == '__main__':
    return_list = []  # 记录每一轮的回报（return）
    total_step = 0
    for i_episode in range(num_episodes):
        state = env.reset()

        for e_i in range(episode_length):
            state_1 = state[0]
            state_2 = state[1]
            state_3 = state[2]
            # 将两个state拼接起来
            state_normal = np.concatenate((state_2, state_3), axis=0)
            a_1 = agent_adversary.take_action(state_1)
            a_2 = agent_normal.take_action(state_normal)
            a_3 = agent_normal.take_action(state_normal)
            # action_normal = [a_2, a_3]
            a_1_one_hot = np.eye(action_dims[0])[a_1]
            a_2_one_hot = np.eye(action_dims[1])[a_2]
            a_3_one_hot = np.eye(action_dims[1])[a_3]
            actions = [a_1_one_hot, a_2_one_hot, a_3_one_hot]
            next_state, reward, done, _ = env.step(actions)
            # 拆分reward
            r_1 = reward[0]
            r_2 = reward[1]
            r_3 = reward[2]
            # reward_normal = [r_2, r_3]
            reward_normal = r_2 + r_3
            # 拆分done
            d_1 = done[0]
            d_2 = done[1]
            d_3 = done[2]
            # done_normal = [d_2, d_3]
            # 有一个done就算done
            done_normal = d_2 or d_3
            # 拆分next_state
            next_state_1 = next_state[0]
            next_state_2 = next_state[1]
            next_state_3 = next_state[2]
            # 将两个state拼接起来
            next_state_normal = np.concatenate((next_state_2, next_state_3), axis=0)


            # 将transition存入replay buffer
            replay_buffer_adversary.add(state_1, a_1, r_1, next_state_1, d_1)
            replay_buffer_normal.add(state_normal, a_2,a_3, reward_normal,
                                     next_state_normal, done_normal)

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

                b_s_2, b_a_2_1,b_a_2_2, b_r_2, b_ns_2, b_d_2 = replay_buffer_normal.sample(batch_size_vdn)
                transition_dict_2 = {
                    'states': b_s_2,
                    'actions1': b_a_2_1,
                    'actions2': b_a_2_2,
                    'next_states': b_ns_2,
                    'rewards': b_r_2,
                    'dones': b_d_2
                }

                
                agent_adversary.update(transition_dict_1)
                agent_normal.update(transition_dict_2)

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
        plt.title(f"{agent_name} by VQN")
        plt.show()
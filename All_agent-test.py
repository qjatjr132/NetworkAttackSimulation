import numpy as np
import gym
from nasim.agents.SAC_agent2 import SAC
from nasim.agents.AC_agent import ACAgent
from nasim.agents.Duel_ddqn_agent import Duel_DDQNAgent
from nasim.agents.ql_replay_agent import TabularQLearningAgent
from nasim.agents.dqn_agent import DQNAgent
from nasim.agents.duel_dqn_agent import DDQNAgent
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

def smooth(data, alpha=0.5):
    smoothed_data = []
    for i, value in enumerate(data):
        if i == 0:
            smoothed_data.append(value)
        else:
            smoothed_value = alpha * value + (1 - alpha) * smoothed_data[i - 1]
            smoothed_data.append(smoothed_value)
    return smoothed_data

plt.rcParams['axes.unicode_minus'] = False
env = gym.make('Small-v0',
               fully_obs=True,
               flat_actions=True,
               flat_obs=True)

ac_agent = ACAgent(env,
                   gamma=0.99,
                   layer=128,
                   ep_steps=1000,
                   lr=0.001,
                   verbose=True,
                   )
#ac_agent.load('model/tiny_ac.pt')
ep_num_list6, ep_result_list6, ep_sd_list6, ep_goal_list6 = ac_agent.train()
#ac_agent.save('model/tiny_ac.pt')

print('---------------------------- mean min max---------------------------------')
print('MEAN : ', np.mean(ep_result_list6))
print('MAX : ', max(ep_result_list6))
print('Min : ', min(ep_result_list6))
print('--------------------------------AC_end------------------------------------')


dqn_agent = DQNAgent(env,
                     render_eval=False,
                     lr=0.001,
                     ep_steps=1000,
                     batch_size=128,
                     replay_size=100000,
                     #final_epsilon=0.00,
                     exploration_steps=10000000,
                     gamma=0.99,
                     hidden_sizes=[1024, 1024],
                     target_update_freq=100000,
                     verbose=True
                     )
#dqn_agent.load('model/tiny_dqn.pt')
ep_num_list3, ep_result_list3, ep_sd_list3, ep_goal_list3 = dqn_agent.train()
#dqn_agent.save('model/tiny_dqn.pt')
print('---------------------------- mean min max---------------------------------')
print('MEAN : ', np.mean(ep_result_list3))
print('MAX : ', max(ep_result_list3))
print('Min : ', min(ep_result_list3))
print('------------------------------DQN_end-----------------------------------')

ddqn_agent = DDQNAgent(env,
                       lr=0.001,
                       ep_steps=1000,
                       batch_size=128,
                       replay_size=100000,
                       #final_epsilon=0.00,
                       exploration_steps=10000000,
                       gamma=0.99,
                       hidden_sizes=[1024, 1024],
                       target_update_freq=100000,
                       verbose=True)

#ddqn_agent.load('model/tiny_ddqn.pt')
ep_num_list2, ep_result_list2, ep_sd_list2, ep_goal_list2 = ddqn_agent.train()
#ddqn_agent.save('model/tiny_ddqn.pt')
print('---------------------------- mean min max---------------------------------')
print('MEAN : ', np.mean(ep_result_list2))
print('MAX : ', max(ep_result_list2))
print('Min : ', min(ep_result_list2))
print('------------------------------DDQN_end-----------------------------------')


ql_agent = TabularQLearningAgent(env,
                                 lr=0.001,
                                 ep_steps=1000,
                                 batch_size=128,
                                 seed=0,
                                 replay_size=100000,
                                 #final_epsilon=0.05,
                                 init_epsilon=1.0,
                                 exploration_steps=100000,
                                 gamma=0.99,
                                 quite=False
                                 )
ep_num_list4, ep_result_list4, ep_sd_list4, ep_goal_list4 = ql_agent.train()
#ql_agent.save('model/tiny_ql.pickle')

print('---------------------------- mean min max---------------------------------')
print('MEAN : ', np.mean(ep_result_list4))
print('MAX : ', max(ep_result_list4))
print('Min : ', min(ep_result_list4))
print('------------------------------QL_end-----------------------------------')


print('------------------------loss, mean plot on-----------------------------')
plt.plot(smooth(ep_num_list6), smooth(ep_result_list6), label='AC')
plt.plot(smooth(ep_num_list4), smooth(ep_result_list4), label='ql')
plt.plot(smooth(ep_num_list3), smooth(ep_result_list3), label='DQN')
plt.plot(smooth(ep_num_list2), smooth(ep_result_list2), label='DDQN')
#plt.title('REWARD / EPISODE')
plt.xlabel('Episode', fontsize=10)
plt.ylabel('Reward', fontsize=10)
plt.legend(loc='best')
plt.show()

plt.plot(smooth(ep_num_list6), smooth(ep_goal_list6), label='AC')
plt.plot(smooth(ep_num_list4), smooth(ep_goal_list4), label='ql')
plt.plot(smooth(ep_num_list3), smooth(ep_goal_list3), label='DQN')
plt.plot(smooth(ep_num_list2), smooth(ep_goal_list2), label='DDQN')
#plt.title('Goal / EPISODE')
plt.xlabel('Episode', fontsize=10)
plt.ylabel('Goal', fontsize=10)
plt.legend(loc='best')
plt.show()

plt.plot(smooth(ep_num_list6), smooth(ep_sd_list6), label='AC')
plt.plot(smooth(ep_num_list4), smooth(ep_sd_list4), label='ql')
plt.plot(smooth(ep_num_list3), smooth(ep_sd_list3), label='DQN')
plt.plot(smooth(ep_num_list2), smooth(ep_sd_list2), label='DDQN')
#plt.title('Train_Step / EPISODE')
plt.xlabel('Episode', fontsize=10)
plt.ylabel('Step', fontsize=10)
plt.legend(loc='best')
plt.show()
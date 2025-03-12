"""A script for training a_a DQN agent and storing best policy """

import nasim
from nasim.agents.attacker.PPO import PPOAgent
import torch
from collections import namedtuple

if __name__ == "__main__":



    env = nasim.make_benchmark('tiny-etri-alp-pps2-r220',
                                             fully_obs=True,
                                             flat_actions=True,
                                             flat_obs=True)

    num_actions = env.action_space.n
    obs_dim = env.A_observation_space.shape[0]

    PPO_agent_attacker = PPOAgent(obs_dim,
                                  num_actions,
                                  lr=0.0003,
                                  gamma=0.01,
                                  lambda_gae=0.95,
                                  entropy_coeff=0.01,
                                  max_grad_norm=0.5,
                                  clip_epsilon=0.2,
                                  K_epochs=10)

    # PPO_agent_attacker.load_model('./policy/ppo_attacker/')
    episode = 0
    episodes = 1000
    max_reward = -1000
    min_steps = episodes

    timestep = 0
    step_limit = 1000
    update_timestep = 64

    while episode < episodes:
        episode += 1
        env.render()
        o, _, _ = env.reset()
        done_a = False

        ak_episode_return = 0
        action_list = list()
        steps = 0
        while (steps < step_limit) and (not done_a):
            timestep +=1
            #attacker
            a_a, a_prob = PPO_agent_attacker.select_action(o)

            next_o_a, r_a, done_a, env_step_limit_reached_a, _ = env.step(a_a, "attacker", mode='None')

            _, value = PPO_agent_attacker.policy_old(torch.FloatTensor(o).unsqueeze(0))
            PPO_agent_attacker.store_transition((o, a_a, a_prob.item(), r_a, done_a, value.item()))

            ak_episode_return += r_a
            o = next_o_a

            steps += 1
            action_list.append(a_a)

            if timestep % update_timestep == 0:
                PPO_agent_attacker.update()

        PPO_agent_attacker.logger.add_scalar("Reward", ak_episode_return, episode)
        PPO_agent_attacker.logger.add_scalar("Success", done_a, episode)
        PPO_agent_attacker.save_model('./policy/ppo_attacker')

        if max_reward <= ak_episode_return:
            max_reward = ak_episode_return
            with open("./action_list/PPO/max_reward_PPO_Action_list.txt", "w") as file:
                # 리스트의 각 요소를 파일에 쓰기
                for item in action_list:
                    file.write(str(item) + ",")

        if min_steps >= steps:
            min_steps = steps
            with open("./action_list/PPO/min_step_PPO_Action_list.txt", "w") as file:
                # 리스트의 각 요소를 파일에 쓰기
                for item in action_list:
                    file.write(str(item) + ",")

        print('\repisode: {}/{} [step: {}] | Attacker [reward: {}, done_a: {}]]'.format(
            episode, episodes, steps, ak_episode_return, done_a), end="", flush=True)

        if episode % 100 == 0:
            print('\n' + '-' * 47 +
                  f'\nepisode: {episode} / {episodes}   [step: {steps}]'
                  f'\nAttacker [reward:{ak_episode_return}, done_a: {done_a}]'
                  + '\n' + '-' * 47)
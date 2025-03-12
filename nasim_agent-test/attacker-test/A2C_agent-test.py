"""A script for training a_a DQN agent and storing best policy """

import nasim
from nasim.agents.attacker.A2C_agent import A2CAgent
import numpy as np

if __name__ == "__main__":



    env = nasim.make_benchmark('tiny-etri-alp-pps2-r220',
                                             fully_obs=True,
                                             flat_actions=True,
                                             flat_obs=True)

    A2C_agent_attacker = A2CAgent(env,
                                  gamma=0.01,
                                  layer=64,
                                  lr=0.0001,
                                  verbose=True,
                                  )

    episode = 0
    episodes = 1000
    while episode < episodes:
        episode += 1
        env.render()
        o, _, _ = env.reset()
        done_a = False
        env_step_limit_reached_a = False

        ak_episode_return = 0

        steps = 0
        step_limit = 1000

        while (steps < step_limit) and (not done_a):
            #attacker
            A2C_agent_attacker.estimate_value(o)

            p = A2C_agent_attacker.Actor(o).cpu().detach().numpy()
            a_a = A2C_agent_attacker.select_action(o)

            e = -np.sum(np.mean(p, dtype=np.float64) * np.log1p(p))
            A2C_agent_attacker.entropy += e

            next_o_a, r_a, done_a, env_step_limit_reached_a, _ = env.step(a_a, "attacker", mode='None')

            ak_episode_return += r_a
            o = next_o_a

            steps += 1
            A2C_agent_attacker.Actor.reward_episode.append(ak_episode_return)

        A2C_agent_attacker.ep_done += 1
        A2C_agent_attacker.finish_episode()

        A2C_agent_attacker.logger.add_scalar("Reward", ak_episode_return, episode)
        A2C_agent_attacker.logger.add_scalar("Success", done_a, episode)

        print('\repisode: {}/{} [step: {}] | Attacker [reward: {}, done_a: {}]]'.format(
            episode, episodes, steps, ak_episode_return, A2C_agent_attacker.env.attacker_goal_reached()), end="", flush=True)

        if episode % 100 == 0:
            print('\n' + '-' * 47 +
                  f'\nepisode: {episode} / {episodes}   [step: {steps}]'
                  f'\nAttacker [reward:{ak_episode_return}, done_a: {str(A2C_agent_attacker.env.attacker_goal_reached())}]'
                  + '\n' + '-' * 47)
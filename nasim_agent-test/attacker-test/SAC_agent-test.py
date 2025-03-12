"""A script for training a_a DQN agent and storing best policy """

import nasim
from nasim.agents.attacker.SAC_agent import SAC

if __name__ == "__main__":

    env = nasim.make_benchmark('tiny-etri-alp-pps2-r220',fully_obs=True, flat_actions=True, flat_obs=True)

    SAC_agent_attacker = SAC(env=env,
                             lr1=3e-4,
                             lr2=3e-4,
                             expl_before=9000,
                             replaybuffer=1000000,
                             training_steps=1000000,
                             batch_size=64,
                             gamma=0.01,
                             alpha=1,
                             verbose=True
                             )


    episode = 0
    episodes = 1000

    max_reward = -1000
    min_steps = episodes

    while episode < episodes:
        action_list = list()

        episode += 1
        o, _, _ = env.reset()
        done_a = False

        env_step_limit_reached_a = False

        ak_episode_return = 0

        steps = 0
        step_limit = 1000

        while (steps < step_limit) and (not done_a):
            #attacker
            if SAC_agent_attacker.steps_done > SAC_agent_attacker.expl_before:
                a_a = SAC_agent_attacker.get_action(o)
            else:
                a_a = SAC_agent_attacker.env.action_space.sample()

            next_o_a, r_a, done_a, env_step_limit_reached_a, _ = SAC_agent_attacker.env.step(a_a, "attacker", mode='None')
            SAC_agent_attacker.replay_buffer.push(o, a_a, r_a, next_o_a, done_a)
            ak_episode_return += r_a
            o = next_o_a
            if len(SAC_agent_attacker.replay_buffer) > SAC_agent_attacker.batch_size:
                SAC_agent_attacker.finsh_ep(SAC_agent_attacker.replay_buffer.sample(SAC_agent_attacker.batch_size))

            SAC_agent_attacker.steps_done += 1
            steps += 1
            action_list.append(a_a)

        if max_reward <= ak_episode_return:
            max_reward = ak_episode_return
            with open("./action_list/SAC/max_reward_SAC_Action_list.txt", "w") as file:
                # 리스트의 각 요소를 파일에 쓰기
                for item in action_list:
                    file.write(str(item) + ",")

        if min_steps >= steps:
            min_steps = steps
            with open("./action_list/SAC/min_step_SAC_Action_list.txt", "w") as file:
                # 리스트의 각 요소를 파일에 쓰기
                for item in action_list:
                    file.write(str(item) + ",")

        SAC_agent_attacker.logger.add_scalar("Reward", ak_episode_return, episode)
        SAC_agent_attacker.logger.add_scalar("Success", done_a, episode)
        SAC_agent_attacker.save_model('./policy/sac_attacker')

        print('\repisode: {}/{} [step: {}] | Attacker [reward: {}, done_a: {}]]'.format(
            episode, episodes, steps, ak_episode_return, SAC_agent_attacker.env.attacker_goal_reached()), end="", flush=True)

        if episode % 100 == 0:
            print('\n' + '-' * 47 +
                  f'\nepisode: {episode} / {episodes}   [step: {steps}]'
                  f'\nAttacker [reward:{ak_episode_return}, done_a: {str(SAC_agent_attacker.env.attacker_goal_reached())}]'
                  + '\n' + '-' * 47)
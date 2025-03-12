"""A script for training a_a DQN agent and storing best policy """

import nasim
from nasim.agents.attacker.double_dqn_agent import DoubleDQNAgent

if __name__ == "__main__":

    env = nasim.make_benchmark('tiny-etri-alpinelab',fully_obs=True, flat_actions=True, flat_obs=True)

    dqn_agent_attacker = DoubleDQNAgent(env, lr=0.001, training_steps=2000000, batch_size=32, replay_size=10000, final_epsilon=0.05,
                                           exploration_steps=5000, gamma=0.01, hidden_sizes=[64, 64], target_update_freq=1000, verbose=True,)

    #dqn_agent_attacker.load('./attacker_policy/dqn_agent')

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
            a_a, a_type, rv, es = dqn_agent_attacker.get_egreedy_action(o, dqn_agent_attacker.get_epsilon(), steps)

            next_o_a, r_a, done_a, env_step_limit_reached_a, _ = dqn_agent_attacker.env.step(a_a, "attacker")
            dqn_agent_attacker.replay.store(o, a_a, next_o_a, r_a, done_a)
            dqn_agent_attacker.steps_done += 1
            loss, mean_v = dqn_agent_attacker.optimize()
            dqn_agent_attacker.logger.add_scalar("attacker_loss", loss, dqn_agent_attacker.steps_done)
            dqn_agent_attacker.logger.add_scalar("attacker_mean_v", mean_v, dqn_agent_attacker.steps_done)

            o = next_o_a
            ak_episode_return += r_a

            steps += 1
        dqn_agent_attacker.logger.add_scalar("Reward", ak_episode_return, episode)
        dqn_agent_attacker.logger.add_scalar("Success", done_a, episode)

        print('\repisode: {}/{} [step: {}] | Attacker [reward: {}, done_a: {}]]'.format(
            episode, episodes, steps, ak_episode_return, dqn_agent_attacker.env.attacker_goal_reached()), end="", flush=True)

        if episode % 100 == 0:
            print('\n' + '-' * 47 +
                  f'\nepisode: {episode} / {episodes}   [step: {steps}]'
                  f'\nAttacker [reward:{ak_episode_return}, done_a: {str(dqn_agent_attacker.env.attacker_goal_reached())}]'
                  + '\n' + '-' * 47)
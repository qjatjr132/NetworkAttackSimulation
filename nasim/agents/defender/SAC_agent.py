import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import nasim as nasim
import numpy as np
import random
from pprint import pprint

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class P_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(P_net, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        action_score = self.fc3(x)
        return F.softmax(action_score, dim=-1)

class Q_net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_net, self).__init__()
        self.fc1 = nn.Linear(state_dim[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        h = torch.sigmoid(self.fc2(h))
        q = self.fc3(h)
        return q

class SAC():
    def __init__(self,
                 env,
                 lr1=1e-3,
                 lr2=1e-4,
                 expl_before=3000,
                 replaybuffer=1000000,
                 training_steps=1000000,
                 batch_size=64,
                 gamma=0.99,
                 alpha=1,
                 verbose=True):
        super(SAC, self).__init__()

        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning SAC with config:")
            pprint(locals())

        self.lr1 = lr1
        self.lr2 = lr2
        self.logger = SummaryWriter('./runs/SAC_defender/')
        self.training_steps = training_steps
        self.expl_before = expl_before
        self.env = env
        self.state_dim = self.env.D_observation_space.shape
        self.action_dim = self.env.defender_action_space.n
        self.gamma = gamma
        self.alpha = alpha
        self.steps_done = 0
        self.replay_buffer = ReplayBuffer(replaybuffer)
        self.batch_size = batch_size

        self.p_net = P_net(self.state_dim, self.action_dim)
        self.q_net = Q_net(self.state_dim, self.action_dim)

        self.loss_fn = torch.nn.MSELoss()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr1)
        self.p_optimizer = torch.optim.Adam(self.p_net.parameters(), lr=self.lr2)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_prob = self.p_net.forward(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item()

    def finsh_ep(self, batch):
        state = batch[0]
        action = batch[1]
        reward = batch[2]
        next_state = batch[3]

        state = torch.from_numpy(state).float().squeeze(1)
        next_state = torch.from_numpy(next_state).float().squeeze(1)
        T = state.size()[0]

        # calculate V
        next_q = self.q_net.forward(next_state)
        next_a_prob = self.p_net.forward(next_state)
        next_v = next_a_prob * (next_q - self.alpha * torch.log(next_a_prob))
        next_v = torch.sum(next_v, 1)

        # train Q
        q = self.q_net.forward(state)
        expect_q = q.clone()
        for i in range(T):
            expect_q[i, action[i]] = reward[i] + self.gamma * next_v[i]
        loss = self.loss_fn(q, expect_q.detach())
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # train Actor
        q = self.q_net.forward(state)
        a_prob = self.p_net.forward(state)
        ploss = a_prob * (self.alpha * torch.log(a_prob) - q)
        ploss = torch.sum(ploss)
        ploss = ploss / T
        self.p_optimizer.zero_grad()
        ploss.backward()
        self.p_optimizer.step()

    def load_model(self, path):
        self.q_net = torch.load(f'{path}/SAC_q_net.pkl')
        self.p_net = torch.load(f'{path}/SAC_p_net.pkl')

    def save_model(self, path):
        torch.save(self.q_net, f'{path}/SAC_q_net.pkl')
        torch.save(self.p_net, f'{path}/SAC_p_net.pkl')

    def train(self):
        ep_num_list = []
        ep_result_list = []
        ep_goal_list = []
        ep_sd_list = []
        ep_sd_list2 = []
        if self.verbose:
            print("\nStarting training")
        num_episodes = 0
        training_steps_remaining = self.training_steps

        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_steps, goal = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar(
                "episode_return", ep_return, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )
            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tsteps done = {self.steps_done} / "
                      f"{self.training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")
                if len(ep_sd_list) == 0:
                    ep_num_list.append(num_episodes)
                    ep_result_list.append(ep_return)
                    ep_sd_list2.append(self.steps_done)
                    ep_sd_list.append(self.steps_done)
                    ep_goal_list.append(goal)
                else:
                    ep_num_list.append(num_episodes)
                    ep_result_list.append(ep_return)
                    ep_sd_list2.append(self.steps_done)
                    ep_sd_list.append(self.steps_done - ep_sd_list2[len(ep_sd_list) - 1])
                    ep_goal_list.append(goal)

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")
        return ep_num_list, ep_result_list, ep_sd_list, ep_goal_list

    def run_train_episode(self, step_limit):
        state = self.env.reset()
        done = False
        env_step_limit_reached = False

        acc_reward = 0  # list to save the true values
        steps = 0

        while not done and not env_step_limit_reached and steps < step_limit:
            if self.steps_done > self.expl_before:
                action = self.get_action(state)
            else:
                action = self.env.action_space.sample()
            #action = self.get_action(state)
            next_state, reward, done, env_step_limit_reached, _ = self.env.step(action)
            acc_reward += reward
            self.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            if len(self.replay_buffer) > self.batch_size:
                self.finsh_ep(self.replay_buffer.sample(self.batch_size))
            if done:
                break
            self.steps_done += 1
            steps += 1

        return acc_reward, steps, self.env.goal_reached()

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.03,
                         render_mode="readable"):
        if env is None:
            env = self.env
        o = env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        line_break = "="*60
        if render:
            print("\n" + line_break)
            print(f"Running EVALUATION using epsilon = {eval_epsilon:.4f}")
            print(line_break)
            env.render(render_mode)
            input("Initial state. Press enter to continue..")

        while not done and not env_step_limit_reached:
            a = self.get_action(o)
            next_o, r, done, env_step_limit_reached, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = {env.action_space.get_action(a)}")
                env.render(render_mode)
                print(f"Reward = {r}")
                print(f"Done = {done}")
                print(f"Step limit reached = {env_step_limit_reached}")
                input("Press enter to continue..")

                if done or env_step_limit_reached:
                    print("\n" + line_break)
                    print("EPISODE FINISHED")
                    print(line_break)
                    print(f"Goal reached = {env.goal_reached()}")
                    print(f"Total steps = {steps}")
                    print(f"Total reward = {episode_return}")

        return episode_return, steps, env.goal_reached()

'''
if __name__ == '__main__':
    env = nasim.make_benchmark('medium',
                               seed=543,
                               fully_obs=True,
                               flat_actions=True,
                               flat_obs=True)

    agent = SAC(env,
                training_steps=2000000,
                verbose=True)

    agent.train()
'''
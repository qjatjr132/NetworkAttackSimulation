import argparse
import gym
import nasim
import numpy as np
from torch.autograd import Variable
from pprint import pprint
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

# Neural Network related attributes
device = torch.device("cuda"
                      if torch.cuda.is_available()
                      else "cpu")

class Actor(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_dim, layers, num_actions):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_dim[0], layers)
        self.linear2 = nn.Tanh()
        self.linear3 = nn.Linear(layers, num_actions)

        # action & reward buffer
        self.policy_history = Variable(torch.Tensor()).to(device)
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = Variable(torch.from_numpy(np.asarray(x)).float().unsqueeze(0)).to(device)
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        action_prob = F.softmax(self.linear3(output), dim=-1)
        return action_prob

    def save_actor(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_actor(self, file_path):
        self.load_state_dict(torch.load(file_path))


class Critic(nn.Module):
    def __init__(self, input_dim, layers):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dim[0], layers)
        self.linear2 = nn.Tanh()
        self.linear3 = nn.Linear(layers, 1)

        self.value_episode = []
        self.value_history = Variable(torch.Tensor()).to(device)

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = Variable(torch.from_numpy(np.asarray(x)).float().unsqueeze(0)).to(device)
        output = F.relu(self.linear1(x))
        output = F.relu(self.linear2(output))
        state_values = self.linear3(output)

        return state_values

    def save_critic(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_critic(self, file_path):
        self.load_state_dict(torch.load(file_path))

class A2CAgent:
    def __init__(self,
                 env,
                 seed=None,
                 gamma=0.99,
                 layer=256,
                 ep_steps=100000,
                 lr = 0.0001,
                 N=10,
                 verbose=True,
                 **kwargs):

        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning A2C with config:")
            pprint(locals())

        self.seed = seed
        self.env = env
        self.layer = layer
        self.lr = lr
        self.gamma = gamma
        self.entropy = 0
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.A_observation_space.shape
        self.ep_steps = ep_steps
        self.ep_done = 0

        self.logger = SummaryWriter('./runs/A2C_attacker/')

        self.Actor = Actor(self.obs_dim, self.layer, self.num_actions).to(device)
        self.Critic = Critic(self.obs_dim, self.layer).to(device)
        self.optimizerA = optim.Adam(self.Actor.parameters(), lr=self.lr)
        self.optimizerC = optim.Adam(self.Critic.parameters(), lr=self.lr)

    def save(self, save_path):
        self.Actor.save_ac(save_path+'Actor')
        self.Critic.save_ac(save_path + 'Critic')

    def load(self, load_path):
        self.Actor.load_ac(load_path+'Actor')
        self.Critic.save_ac(load_path + 'Critic')

    def estimate_value(self, state):
        pred = self.Critic(state).squeeze(0)
        if self.Critic.value_history.dim() != 0:
            self.Critic.value_history = torch.cat([self.Critic.value_history, pred])
        else:
            self.Critic.policy_history = (pred)

    def select_action(self, state):
        probs = self.Actor(state)
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # place log probabilities into the policy history log\pi(a | s)
        if self.Actor.policy_history.dim() != 0:
            self.Actor.policy_history = torch.cat([self.Actor.policy_history, m.log_prob(action)])
        else:
            self.Actor.policy_history = m.log_prob(action)

        # the action to take (left or right)
        return action.item()

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        q_val= []

        # calculate the true value using rewards returned from the environment
        for r in self.Actor.reward_episode[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            q_val.insert(0, R)

        q_vals = torch.FloatTensor(q_val).to(device)
        values = self.Critic.value_history
        log_probs = self.Actor.policy_history

        # print(values)
        # print(log_probs)
        advantage = q_vals - values

        self.optimizerC.zero_grad()
        critic_loss = 0.0005 * advantage.pow(2).mean()
        critic_loss.backward()
        self.optimizerC.step()

        self.optimizerA.zero_grad()
        actor_loss = (-log_probs * advantage.detach()).mean() + 0.001 * self.entropy
        actor_loss.backward()
        self.optimizerA.step()

        self.Actor.reward_episode = []
        self.Actor.policy_history = Variable(torch.Tensor()).to(device)
        self.Critic.value_history = Variable(torch.Tensor()).to(device)

        return actor_loss, critic_loss

    def train(self):
        ep_num_list = []
        ep_result_list = []
        ep_goal_list = []
        ep_sd_list = []
        ep_time = 0

        if self.verbose:
            print("\nStarting training")
        num_episodes = 0

        while self.ep_done < self.ep_steps:
            ep_results = self.run_train_episode()
            ep_return, ep_steps, goal, total_time = ep_results
            num_episodes += 1

            self.logger.add_scalar("episode", num_episodes, self.ep_done)
            self.logger.add_scalar(
                "episode_return", ep_return, self.ep_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.ep_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.ep_done
            )
            ep_num_list.append(num_episodes)
            ep_result_list.append(ep_return)
            ep_sd_list.append(ep_steps)
            ep_goal_list.append(goal)
            ep_time += total_time

            if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode {num_episodes}:")
                print(f"\tep done = {self.ep_done} / "
                      f"{self.ep_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")
                print(f"\ttime = {total_time}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.ep_done} / {self.ep_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")
            print(f"\ttotal_time = {ep_time}")
        return ep_num_list, ep_result_list, ep_sd_list, ep_goal_list

    def run_train_episode(self):
        time_start = time()
        o = self.env.reset()
        done = False
        env_step_limit_reached = False

        episode_return = 0  # list to save the true values
        steps = 0

        while not done and not env_step_limit_reached:
            self.estimate_value(o)

            p = self.Actor(o).cpu().detach().numpy()
            a = self.select_action(o)

            e = -np.sum(np.mean(p, dtype=np.float64) * np.log1p(p))
            self.entropy += e

            next_o, r, done, env_step_limit_reached, _ = self.env.step(a)

            o = next_o
            episode_return += r
            steps += 1
            self.Actor.reward_episode.append(episode_return)
        # perform backprop
        self.ep_done += 1
        self.finish_episode()
        time_end = time()
        total_time = (time_end - time_start)
        return episode_return, steps, self.env.goal_reached(), total_time

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
            a = self.select_action(o)
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument("env_name", type=str, default='tiny', help="benchmark scenario name")
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=True,
                               flat_actions=True,
                               flat_obs=True)
    ac_agent = A2CAgent(env,
                        verbose=args.quite,
                        **vars(args))
    ac_agent.train()

if __name__ == '__main__':
    main()
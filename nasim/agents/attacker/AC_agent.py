import argparse
import gym
import nasim
import numpy as np
from itertools import count
from collections import namedtuple
from pprint import pprint
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, input_dim, layers, num_actions):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_dim[0], layers)

        # actor's layer
        self.action_head = nn.Linear(layers, num_actions)

        # critic's layer
        self.value_head = nn.Linear(layers, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

    def save_ac(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_ac(self, file_path):
        self.load_state_dict(torch.load(file_path))
class ACAgent:
    def __init__(self,
                 env,
                 seed=None,
                 gamma=0.99,
                 layer=128,
                 ep_steps=500000,
                 lr=3e-2,
                 verbose=True,
                 **kwargs):

        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning AC with config:")
            pprint(locals())

        self.seed = seed
        self.env = env
        self.layer = layer
        self.lr = lr
        self.gamma = gamma

        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape
        self.ep_steps = ep_steps
        self.ep_done = 0
        self.step = 0

        self.logger = SummaryWriter()

        # Neural Network related attributes
        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.AC = Policy(self.obs_dim, self.layer, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.AC.parameters(), lr=self.lr)
        self.eps = np.finfo(np.float32).eps.item()

    def save(self, save_path):
        self.AC.save_ac(save_path)

    def load(self, load_path):
        self.AC.load_ac(load_path)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs, state_value = self.AC(state)
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.AC.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.AC.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.AC.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.AC.rewards[:]
        del self.AC.saved_actions[:]

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
            ep_return, ep_steps, total_step, goal, total_time = ep_results
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
                print(f"\taccumulate_step = {total_step}")

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.ep_done} / {self.ep_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")
            print(f"\ttotal_time = {ep_time}")
            print(f"\ttotal_step = {total_step}")
        return ep_num_list, ep_result_list, ep_sd_list, ep_goal_list

    def run_train_episode(self):
        time_start = time()
        o = self.env.reset()
        done = False
        env_step_limit_reached = False

        episode_return = 0  # list to save the true values
        steps = 0

        while not done and not env_step_limit_reached:
            a = self.select_action(o)
            next_o, r, done, env_step_limit_reached, _ = self.env.step(a)

            self.AC.rewards.append(r)
            o = next_o
            episode_return += r
            steps += 1
            self.step += 1
        # perform backprop
        self.ep_done += 1
        self.finish_episode()
        time_end = time()
        total_time = (time_end - time_start)
        return episode_return, steps, self.step, self.env.goal_reached(), total_time

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
    ac_agent = ACAgent(env,
                       verbose=args.quite,
                       **vars(args))
    ac_agent.train()

if __name__ == '__main__':
    main()
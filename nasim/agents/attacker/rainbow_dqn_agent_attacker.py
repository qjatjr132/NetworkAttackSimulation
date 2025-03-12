import random
import math
from pprint import pprint

from gym import error
import numpy as np

from tqdm import tqdm
import datetime as dt

import nasim

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    raise error.DependencyNotInstalled(
        f"{e}. (HINT: you can install dqn_agent dependencies by running "
        "'pip install nasim_with_defender[dqn]'.)"
    )


LINE_BREAK = "-"*60
LINE_BREAK2 = "="*60
def print_actions(action_space):
    for a in range(action_space.n):
        print(f"{a} {action_space.get_action(a)}")
    print(LINE_BREAK)

# class ReplayMemory:

#     def __init__(self, capacity, s_dims, device="cpu"):
#         self.capacity = capacity
#         self.device = device
#         self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
#         self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
#         self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
#         self.r_buf = np.zeros(capacity, dtype=np.float32)
#         self.done_buf = np.zeros(capacity, dtype=np.float32)
#         self.ptr, self.size = 0, 0

#         self.dup_count = 0
#         self.sar_dict = dict()
#         self.sar_idx = 0

#     def store(self, s, a, next_s, r, done):
#         self.s_buf[self.ptr] = s
#         self.a_buf[self.ptr] = a
#         self.next_s_buf[self.ptr] = next_s
#         self.r_buf[self.ptr] = r
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.capacity
#         self.size = min(self.size+1, self.capacity)


#     def store_new_trans(self, s, a, next_s, r, done):

#         if self.ptr != 0:
#             for i in range(self.ptr):
#                 if (np.array_equal(self.s_buf[i], s) == True) and \
#                     self.a_buf[i] == a and \
#                     (np.array_equal(self.next_s_buf[i], next_s) == True) and \
#                     self.r_buf[i] == r and \
#                     self.done_buf[i] == done:
#                     self.dup_count += 1
#                     print(f"[+] [# {self.dup_count}] same transition(s, a, ns, r, d) data :")
#                     print(f"[+] s={np.where(s != 0)[0]}, a=[{a}], next_s={np.where(next_s != 0)[0]}, r={r}, done={done}")
#                     return
#                 else:
#                     continue

#         self.s_buf[self.ptr] = s
#         self.a_buf[self.ptr] = a
#         self.next_s_buf[self.ptr] = next_s
#         self.r_buf[self.ptr] = r
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.capacity
#         self.size = min(self.size + 1, self.capacity)
#         return
    
#     def store_new_data(self, s, a, next_s, r, done):
#         if isinstance(s, np.ndarray):
#             s_str = str(s.astype(np.int32))
#         sar = s_str + '_' + str(a) + '_' + str(r)
#         if r == -1 and sar in self.sar_dict:
#             self.dup_count += 1
#             return
#         else:
#             self.sar_dict[sar] = self.sar_idx
#             self.sar_idx += 1
#             print(f"[+] state_action_reward[{self.ptr}] : {sar[-20:]}")

#             self.s_buf[self.ptr] = s
#             self.a_buf[self.ptr] = a
#             self.next_s_buf[self.ptr] = next_s
#             self.r_buf[self.ptr] = r
#             self.done_buf[self.ptr] = done
#             self.ptr = (self.ptr + 1) % self.capacity
#             self.size = min(self.size + 1, self.capacity)
#         return

#     def sample_batch(self, batch_size, steps_done):
#         PRINT_ON = False
#         log_print_count = 200

#         sample_idxs = np.random.choice(self.size, batch_size)
#         batch = [self.s_buf[sample_idxs],
#                  self.a_buf[sample_idxs],
#                  self.next_s_buf[sample_idxs],
#                  self.r_buf[sample_idxs],
#                  self.done_buf[sample_idxs]]

#         if PRINT_ON:
#             self._print_s_buf_batch(self.s_buf[sample_idxs],
#                                     self.a_buf[sample_idxs],
#                                     self.r_buf[sample_idxs],
#                                     self.next_s_buf[sample_idxs],
#                                     sample_idxs)

#         return [torch.from_numpy(buf).to(self.device) for buf in batch]

#     def _print_s_buf_batch(self, s_buf_batch, a_buf_batch, r_buf_batch, ns_buf_batch, sample_idxs):

#         depth = s_buf_batch.shape[0]

#         s_buf_batch_nonzero_index = list()
#         ns_buf_batch_nonzero_index = list()

#         for d_pos in range(depth):
#             state = s_buf_batch[d_pos]
#             state_nonzero_index = list()
#             for s_pos in range(state.shape[0]):
#                 if state[s_pos] != 0:
#                     state_nonzero_index.append(s_pos)
#             s_buf_batch_nonzero_index.append(state_nonzero_index)

#             nstate = ns_buf_batch[d_pos]
#             nstate_nonzero_index = list()
#             for s_pos in range(nstate.shape[0]):
#                 if nstate[s_pos] != 0:
#                     nstate_nonzero_index.append(s_pos)
#             ns_buf_batch_nonzero_index.append(nstate_nonzero_index)

#         for i in range(len(s_buf_batch_nonzero_index)):

#             if r_buf_batch[i] != float(-1.0):
#                 print(f"[+] A[{a_buf_batch[i]}], R[{r_buf_batch[i]}], "
#                       f"\n\t s_b_nz[{i}] {s_buf_batch_nonzero_index[i]}, "
#                       f"\n\tns_b_nz[{i}] {ns_buf_batch_nonzero_index[i]}")

#                 print(f"[+] sample_idxs : {[ sample_idxs[i] for i in range(len(sample_idxs)) ]}")
#         return


#     def _print_s_buf(self, s_buf):
#         print(f"[+] _print_s_buf: _s_buf (100000, 174) :")
#         depth = min(s_buf.shape[0], self.size)

#         s_buf_nonzero_index = list()

#         for d_pos in range(depth):
#             state = s_buf[d_pos]
#             state_nonzero_index = list()
#             for s_pos in range(state.shape[0]):
#                 if state[s_pos] != 0:
#                     state_nonzero_index.append(s_pos)
#             s_buf_nonzero_index.append(state_nonzero_index)

#         for i in range(len(s_buf_nonzero_index)):
#             print(f"[+] s_buf_nonzero_index : [{i}] {s_buf_nonzero_index[i]}")
#         return

#     def _print_a_buf(self, a_buf):
#         print(f"\n[+] _print_a_buf: _s_buf (100000, 1) : ")
#         depth = min(a_buf.shape[0], self.size)

#         a_buf_index = list()

#         for d_pos in range(depth):
#             action = a_buf[d_pos][0]
#             a_buf_index.append(action)
#         print(f"[+] a_buf_index :\n{a_buf_index}")
#         return
    
class PrioritizedReplayMemory:
    def __init__(self, capacity, s_dims, alpha=0.6, beta=0.4, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, s, a, next_s, r, done, priority=1.0):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.priorities[self.ptr] = priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size):
        if self.size == 0:
            return None

        scaled_priorities = np.power(self.priorities[:self.size], self.alpha)
        prob = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(self.size, batch_size, p=prob)

        batch = [self.s_buf[indices],
                 self.a_buf[indices],
                 self.next_s_buf[indices],
                 self.r_buf[indices],
                 self.done_buf[indices],
                 indices, prob[indices]]

        return [torch.from_numpy(buf).to(self.device) for buf in batch]

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.ger(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
class Dueling_DQN(nn.Module):
    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        self.fc1 = nn.Linear(input_dim[0], layers[0])
        self.fc2 = nn.Linear(layers[0], layers[0])
        self.V = NoisyLinear(layers[0], 1)
        self.A = NoisyLinear(layers[0], num_actions)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)
        A = self.A(s)
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))
        return Q

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x, steps_done, log_print_count, PRINT_ON):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)

            fwd_x = self.forward(x)

            if steps_done % log_print_count == 0 & PRINT_ON == True:
                print(f"[+] @get_action: fwd_x(q_vals-DQN out) :{fwd_x[0][0:5]}, max:{fwd_x.max(1)[0]}")
            return fwd_x.max(1)[1], fwd_x

class DQNAgent_Attacker:
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=20000,
                 pre_training_steps=50000,
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 verbose=True,
                 **kwargs):

        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning DQN with config:")
            pprint(locals())

        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        self.env = env
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.A_observation_space.shape

        self.logger = SummaryWriter('./runs/rainbow_dqn_attacker/')

        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0,
                                            self.final_epsilon,
                                            self.exploration_steps)
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0

        self.pre_training_steps = pre_training_steps
        self.pre_steps_done = 0

        self.device = torch.device("cuda"
                                   if torch.cuda.is_available()
                                   else "cpu")

        self.dqn = Dueling_DQN(self.obs_dim,
                               hidden_sizes,
                               self.num_actions).to(self.device)
        if self.verbose:
            print(f"\nUsing Neural Network running on device={self.device}:")
            print(self.dqn)

        self.target_dqn = Dueling_DQN(self.obs_dim,
                                      hidden_sizes,
                                      self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = PrioritizedReplayMemory(replay_size,
                                              self.obs_dim,
                                              device=self.device)

        self.log_print_count = 200
        self.PRINT_ON = False

        self.pre_episode_return = -1000
        self.hold_epsilon = False
        self.ep_pos_done = 0

    def save(self, save_path):
        self.dqn.save_DQN(save_path)

    def load(self, load_path):
        self.dqn.load_DQN(load_path)

    def get_epsilon(self):
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon

    def get_epsilon_adaptive(self):
        if self.ep_pos_done < self.exploration_steps:
            return self.epsilon_schedule[self.ep_pos_done]
        return self.final_epsilon

    def get_egreedy_action(self, o, epsilon, steps):
        random_value = random.random()

        if random_value > epsilon:
            if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
                print(f"[+] @get_egreedy_action: rand#>e={epsilon} (exploitation), steps=[{steps}], o(in-state) : {o[0:15]}")
            o = torch.from_numpy(o).float().to(self.device)

            action_tensor, Q_val = self.dqn.get_action(o, self.steps_done, self.log_print_count, self.PRINT_ON)
            action = action_tensor.cpu().item()
            return action, 'Q', random_value, epsilon
        else:
            if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
                print(f"[+] @get_egreedy_action: rand#<e={epsilon} (exploration), steps=[{steps}], o(in-state) : {o[0:15]}")
            return random.randint(0, self.num_actions - 1), 'R', random_value, epsilon

    def optimize(self):
        if self.replay.size < self.batch_size:
            return 0.0, 0.0

        batch = self.replay.sample_batch(self.batch_size)
        if batch is None:
            return 0.0, 0.0
        
        s_batch, a_batch, next_s_batch, r_batch, d_batch, indices, prob = batch

        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()
        _, a_prime = q_vals_raw.max(1)

        if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
            print(f"[+] q_vals : {q_vals}")

        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_vals = target_q_val_raw.max(1)[0]
            y = r_batch + self.discount * target_q_vals * (1 - d_batch)

        loss = self.loss_fn(q_vals, y)
        loss = (loss * prob).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities
        new_priorities = np.abs(q_vals.cpu().detach().numpy() - y.cpu().detach().numpy()) + 1e-5
        self.replay.update_priorities(indices, new_priorities)
        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()

        return loss.item(), mean_v

    def pre_train(self):
        if self.verbose:
            print("\nStarting pre-training")

        num_episodes = 0
        pre_training_steps_remaining = self.pre_training_steps

        while self.pre_steps_done < self.pre_training_steps:

            ep_results = self.run_pre_train_episode(pre_training_steps_remaining)

            ep_return, ep_steps, goal, ep_reward_list, ep_action_list, \
                ep_actres_list, ep_action_type_list, ep_ranval_list, \
                ep_epsilon_list = ep_results

            num_episodes += 1
            pre_training_steps_remaining -= ep_steps

            if True:

                print(f"\nEpisode(pre-train) {num_episodes}:")
                print(f"\tpre_steps done = {self.pre_steps_done} / "
                      f"{self.pre_training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")

                print(f"\tepsilon = {self.get_epsilon()}")
                print(f"\treward_list[{len(ep_reward_list)}] = {ep_reward_list}")
                print(f"\taction_list[{len(ep_action_list)}] = {ep_action_list}")
                print(f"\tactype_list[{len(ep_action_type_list)}] = {ep_action_type_list}")
                print(f"\tranvdl_list[{len(ep_ranval_list)}] = {[round(rv, 3) for rv in ep_ranval_list]}")
                print(f"\tepsiln_list[{len(ep_epsilon_list)}] = {[round(rv, 3) for rv in ep_epsilon_list]}")
                print(f"\tactres_list[{len(ep_actres_list)}] = {ep_actres_list}")

    def train(self):
        if self.verbose:
            print(LINE_BREAK)
            print("Starting training")
            print(LINE_BREAK)
            print("Action space")
            print(LINE_BREAK)
            print_actions(self.env.action_space)
            print(LINE_BREAK)
            print("State space")
            print(LINE_BREAK)
            self.env.render("readable", self.env.reset())
            print(LINE_BREAK2)

        num_episodes = 0
        training_steps_remaining = self.training_steps


        pbar = tqdm(desc='training_steps', total=self.training_steps)
        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)

            ep_return, ep_steps, goal, ep_reward_list, ep_action_list, ep_actres_list, ep_action_type_list, ep_ranval_list, ep_epsilon_list, ep_obs_list = ep_results

            num_episodes += 1
            training_steps_remaining -= ep_steps

            self.logger.add_scalar("episode", num_episodes, self.steps_done)
            self.logger.add_scalar(
                "epsilon", self.get_epsilon(), self.steps_done
            )
            self.logger.add_scalar(
                "episode_return", ep_return, self.steps_done
            )
            self.logger.add_scalar(
                "episode_steps", ep_steps, self.steps_done
            )
            self.logger.add_scalar(
                "episode_goal_reached", int(goal), self.steps_done
            )

            if num_episodes < 100:
                if num_episodes % 10 == 0 and self.verbose:
                    print(f"\nEpisode {num_episodes}:")
                    print(f"\tsteps done = {self.steps_done} / "
                          f"{self.training_steps}")
                    print(f"\treturn = {ep_return}")
                    print(f"\tgoal = {goal}")

                    print(f"\tepsilon = {self.get_epsilon()}")

                    PRINT_DQN_PARAMS = False
                    if PRINT_DQN_PARAMS:
                        self.print_dqn_parameters()
            else:
                if num_episodes % 50 == 0 and self.verbose:
                    print(f"\nEpisode {num_episodes}:")
                    print(f"\tsteps done = {self.steps_done} / "
                          f"{self.training_steps}")
                    print(f"\treturn = {ep_return}")
                    print(f"\tgoal = {goal}")

                    print(f"\tepsilon = {self.get_epsilon()}")

                    PRINT_DQN_PARAMS = False
                    if PRINT_DQN_PARAMS:
                        self.print_dqn_parameters()

                if num_episodes == 399 and self.verbose:
                    print(f"\nEpisode {num_episodes}:")
                    print(f"\tsteps done = {self.steps_done} / "
                          f"{self.training_steps}")
                    print(f"\treturn = {ep_return}")
                    print(f"\tgoal = {goal}")

                    print(f"\tepsilon = {self.get_epsilon()}")

                    render_mode = "readable"
                    for i in range(len(ep_action_list)):
                        self.env.render(render_mode, ep_obs_list[i])
                        action = self.env.action_space.get_action(ep_action_list[i])
                        print(f"Performing[{i}]: [a:{ep_action_list[i]}] {action}")
                    self.env.render(render_mode, ep_obs_list[-1])

                    PRINT_DQN_PARAMS = False
                    if PRINT_DQN_PARAMS:
                        self.print_dqn_parameters()

            pbar.update(ep_steps)
        pbar.close()

        self.logger.close()
        if self.verbose:
            print("Training complete")
            print(f"\nEpisode {num_episodes}:")
            print(f"\tsteps done = {self.steps_done} / {self.training_steps}")
            print(f"\treturn = {ep_return}")
            print(f"\tgoal = {goal}")

    def print_dqn_parameters(self):
        print(f"\nDQN-weight: \n{self.dqn.layers[0].weight}")
        print(f"\nDQN-weight-grad: \n{self.dqn.layers[0].weight.grad}")
        print(f"===============")
        return

    def run_pre_train_episode(self, step_limit):
        o = self.env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        reward_list = list()
        action_list = list()
        actres_list = list()
        action_type_list = list()
        ranval_list = list()
        epsilon_list = list()

        while not done and not env_step_limit_reached and steps < step_limit:

            a = random.randint(0, self.num_actions-1)
            next_o, r, done, env_step_limit_reached, _ = self.env.step(a)

            if r > 0:
                self.replay.store_new_data(o, a, next_o, r, done)
            self.pre_steps_done += 1

            o = next_o
            episode_return += r
            steps += 1

            reward_list.append(r)
            action_list.append(a)
            actres_list.append(_['success'])
            action_type_list.append('R')
            ranval_list.append(0.01)
            epsilon_list.append(0.05)

        return episode_return, steps, self.env.goal_reached(), reward_list, \
            action_list, actres_list, action_type_list, ranval_list, epsilon_list

    def run_train_episode(self, step_limit):
        o = self.env.reset()
        done = False
        env_step_limit_reached = False

        steps = 0
        episode_return = 0

        reward_list = list()
        action_list = list()
        actres_list = list()
        action_type_list = list()
        ranval_list = list()
        epsilon_list = list()
        obs_list = list()
        obs_list.append(o)

        while not done and not env_step_limit_reached and steps < step_limit:

            a, a_type, rv, es = self.get_egreedy_action(o, self.get_epsilon(), steps)

            if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
                print(f"[+] steps=[{steps}], action=[{a}]")

            next_o, r, done, env_step_limit_reached, _ = self.env.step(a)
            self.replay.store(o, a, next_o, r, done)
            self.steps_done += 1
            loss, mean_v = self.optimize()
            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("mean_v(q_vals)", mean_v, self.steps_done)
            o = next_o
            episode_return += r
            steps += 1
            reward_list.append(r)
            action_list.append(a)
            actres_list.append(_['success'])
            action_type_list.append(a_type)
            ranval_list.append(rv)
            epsilon_list.append(es)
            obs_list.append(o)
        return episode_return, steps, self.env.goal_reached(), reward_list, action_list, actres_list, action_type_list, ranval_list, epsilon_list, obs_list

    def run_eval_episode(self,
                         env=None,
                         render=False,
                         eval_epsilon=0.05,
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
            a, a_type, rv, es = self.get_egreedy_action(o, eval_epsilon, steps)
            next_o, r, done, env_step_limit_reached, _ = env.step(a)
            o = next_o
            episode_return += r
            steps += 1
            if render:
                print("\n" + line_break)
                print(f"Step {steps}")
                print(line_break)
                print(f"Action Performed = [a:{a}] {env.action_space.get_action(a)}")
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    parser.add_argument("--render_eval", action="store_true",
                        help="Renders final policy")
    parser.add_argument("-o", "--partially_obs", action="store_true",
                        help="Partially Observable Mode")
    parser.add_argument("--hidden_sizes", type=int, nargs="*",
                        default=[128],

                        help="(default=[64, 64])")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=40000,
                        help="training steps (default=50000)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="(default=32)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="(default=1000)")
    parser.add_argument("--seed", type=int, default=386,
                        help="(default=0)")
    parser.add_argument("--replay_size", type=int, default=100000,
                        help="(default=100000)")
    parser.add_argument("--final_epsilon", type=float, default=0.05,
                        help="(default=0.05)")
    parser.add_argument("--init_epsilon", type=float, default=1.0,
                        help="(default=1.0)")
    parser.add_argument("--exploration_steps", type=int, default=10000,
                        help="(default=10000)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="(default=0.99)")
    parser.add_argument("--quite", action="store_false",
                        help="Run in Quite mode")
    args = parser.parse_args()

    env = nasim.make_benchmark(args.env_name,
                               args.seed,
                               fully_obs=not args.partially_obs,
                               flat_actions=True,
                               flat_obs=True)

    dqn_agent = DQNAgent_Attacker(env, verbose=args.quite, **vars(args))



    dqn_agent.train()


    date = dt.datetime.now()
    file_path_date = "./etri_dqn_policies/dqn_agent" + date.strftime("-%y_%m%d_%H%M") + ".pt"
    file_path = "./etri_dqn_policies/dqn_agent.pt"
    dqn_agent.save(file_path_date)
    dqn_agent.save(file_path)

    dqn_agent.run_eval_episode(render=args.render_eval)


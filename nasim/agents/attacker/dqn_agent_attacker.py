"""An example DQN Agent.

It uses pytorch 1.5+ and tensorboard libraries (HINT: these dependencies can
be installed by running pip install nasim_with_defender[dqn])

To run 'tiny' benchmark scenario with default settings, run the following from
the nasim_with_defender/agents dir:

$ python dqn_agent.py tiny

To see detailed results using tensorboard:

$ tensorboard --logdir runs/

To see available hyperparameters:

$ python dqn_agent.py --help

Notes
-----

This is by no means a state of the art implementation of DQN, but is designed
to be an example implementation that can be used as a reference for building
your own agents.
"""
import random
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


# kjkoo : (23-1024) : debugging 용도
LINE_BREAK = "-"*60
LINE_BREAK2 = "="*60
def print_actions(action_space):
    for a in range(action_space.n):
        print(f"{a} {action_space.get_action(a)}")
    print(LINE_BREAK)

class ReplayMemory:

    def __init__(self, capacity, s_dims, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
        # kjkoo
        self.dup_count = 0
        self.sar_dict = dict()      # state action reward
        self.sar_idx = 0

    def store(self, s, a, next_s, r, done):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    # kjkoo : (23-0718)
    #   - 새로운 transitions에 해당되는 데이터만 버퍼에 저장한다.
    def store_new_trans(self, s, a, next_s, r, done):
        # v1
        # if self.ptr != 0:
        #     for i in range(self.ptr):
        #         if np.array_equal(self.s_buf[i], s) == False:       # 다르면 저장
        #             #print(f"[+] (self.s_buf[i] == s).all() : {(self.s_buf[i] == s).all()}")
        #             break
        #         else:
        #             #print(f"[+] self.s_buf[i]={np.where(self.s_buf[i] != 0)[0]}, s={np.where(s != 0)[0]}")
        #             if self.a_buf[i] != a:
        #                 break
        #             else:
        #                 if np.array_equal(self.next_s_buf[i], next_s) == False:     # 다르면 저장
        #                     break
        #                 else:
        #                     if self.r_buf[i] != r:
        #                         break
        #                     else:
        #                         if self.done_buf[i] != done:
        #                             break
        #                         else:
        #                             # 모든 데이터가 같은 경우
        #                             print(f"[+] s={np.where(s != 0)[0]}, a=[{a}], next_s={np.where(next_s != 0)[0]}, r={r}, done={done}")
        #                             return

        # v2
        if self.ptr != 0:
            for i in range(self.ptr):
                if (np.array_equal(self.s_buf[i], s) == True) and \
                    self.a_buf[i] == a and \
                    (np.array_equal(self.next_s_buf[i], next_s) == True) and \
                    self.r_buf[i] == r and \
                    self.done_buf[i] == done:
                    self.dup_count += 1
                    print(f"[+] [# {self.dup_count}] same transition(s, a, ns, r, d) data :")
                    print(f"[+] s={np.where(s != 0)[0]}, a=[{a}], next_s={np.where(next_s != 0)[0]}, r={r}, done={done}")
                    return
                else:
                    #print(f"[+]buf_i={i}")
                    continue

        # kjkoo : (23-0805) - v3
        #   - 입력을 string으로 변환. concatenation 후에 리스트로 저장. 새로운 입력이 리스트에 없으면 저장

        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return

    # kjkoo : (23-0806)
    #   - 새로운 transitions에 해당되는 데이터만 버퍼에 저장한다. (그렇게 하지 않아도 될 것 같긴 한데..)
    #   - (제안1) 우선 순위가 높은 데이터를 먼저 학습...???

    def store_new_data(self, s, a, next_s, r, done):

        # kjkoo : (23-0806)
        #   - 공격 성공한 action 확인
        #if r > 0:
        #    print(f"[+] RM-ptr={self.ptr}, a={a}, r={r}, d={done}") #, s={s}")

        # kjkoo : (23-0807)
        #   - 공격 실패한 중복된 패턴은 저장하지 않는다.
        if isinstance(s, np.ndarray):
            s_str = str(s.astype(np.int32))   # origin : np.int (23-0701)
        sar = s_str + '_' + str(a) + '_' + str(r)

        if r == -1 and sar in self.sar_dict:
            # 공격이 실패하고, 이미 메모리에 있는 패턴이면 스킵
            self.dup_count += 1
            #print(f"[+] dup_count = {self.dup_count} ")
            return
        else:
            # 새로운 state_action_reward(sar) 패턴이면 replay 메모리에 저장
            #if sar not in self.sar_dict:
            self.sar_dict[sar] = self.sar_idx
            self.sar_idx += 1
            print(f"[+] state_action_reward[{self.ptr}] : {sar[-20:]}")

            # replay memory에 저장
            self.s_buf[self.ptr] = s
            self.a_buf[self.ptr] = a
            self.next_s_buf[self.ptr] = next_s
            self.r_buf[self.ptr] = r
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        return

    def sample_batch(self, batch_size, steps_done):
        PRINT_ON = False
        log_print_count = 200

        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [self.s_buf[sample_idxs],
                 self.a_buf[sample_idxs],
                 self.next_s_buf[sample_idxs],
                 self.r_buf[sample_idxs],
                 self.done_buf[sample_idxs]]

        # kjkoo : (23-0711) debug
        # self._print_s_buf(self.s_buf)
        # self._print_a_buf(self.a_buf)
        # if steps_done % log_print_count == 0 & PRINT_ON:
        if PRINT_ON:
            self._print_s_buf_batch(self.s_buf[sample_idxs],
                                    self.a_buf[sample_idxs],
                                    self.r_buf[sample_idxs],
                                    self.next_s_buf[sample_idxs],
                                    sample_idxs)

        return [torch.from_numpy(buf).to(self.device) for buf in batch]

    # kjkoo : (23-0711)
    def _print_s_buf_batch(self, s_buf_batch, a_buf_batch, r_buf_batch, ns_buf_batch, sample_idxs):
        #print(f"[+] _print_s_buf_batch: _s_buf_batch (32, 174) :")
        depth = s_buf_batch.shape[0]
        #len_state = s_buf.shape[1]
        s_buf_batch_nonzero_index = list()
        ns_buf_batch_nonzero_index = list()

        for d_pos in range(depth):
            state = s_buf_batch[d_pos]
            state_nonzero_index = list()
            for s_pos in range(state.shape[0]):
                if state[s_pos] != 0:
                    state_nonzero_index.append(s_pos)
            s_buf_batch_nonzero_index.append(state_nonzero_index)

            nstate = ns_buf_batch[d_pos]
            nstate_nonzero_index = list()
            for s_pos in range(nstate.shape[0]):
                if nstate[s_pos] != 0:
                    nstate_nonzero_index.append(s_pos)
            ns_buf_batch_nonzero_index.append(nstate_nonzero_index)

        # batch size만큼 출력
        for i in range(len(s_buf_batch_nonzero_index)):
            #print(f"[+] _print_s_buf_batch: _s_buf_batch (32, 174) :")
            if r_buf_batch[i] != float(-1.0):
                print(f"[+] A[{a_buf_batch[i]}], R[{r_buf_batch[i]}], "
                      f"\n\t s_b_nz[{i}] {s_buf_batch_nonzero_index[i]}, "
                      f"\n\tns_b_nz[{i}] {ns_buf_batch_nonzero_index[i]}")
                #print(f"[+] sample_idxs : {sample_idxs}")
                print(f"[+] sample_idxs : {[ sample_idxs[i] for i in range(len(sample_idxs)) ]}")
        return



    def _print_s_buf(self, s_buf):
        print(f"[+] _print_s_buf: _s_buf (100000, 174) :")
        depth = min(s_buf.shape[0], self.size)
        #len_state = s_buf.shape[1]
        s_buf_nonzero_index = list()

        for d_pos in range(depth):
            state = s_buf[d_pos]
            state_nonzero_index = list()
            for s_pos in range(state.shape[0]):
                if state[s_pos] != 0:
                    state_nonzero_index.append(s_pos)
            s_buf_nonzero_index.append(state_nonzero_index)

        for i in range(len(s_buf_nonzero_index)):
            print(f"[+] s_buf_nonzero_index : [{i}] {s_buf_nonzero_index[i]}")
        return

    def _print_a_buf(self, a_buf):
        print(f"\n[+] _print_a_buf: _s_buf (100000, 1) : ")
        depth = min(a_buf.shape[0], self.size)
        #len_state = s_buf.shape[1]
        a_buf_index = list()

        for d_pos in range(depth):
            action = a_buf[d_pos][0]
            a_buf_index.append(action)
        print(f"[+] a_buf_index :\n{a_buf_index}")
        return

class DQN(nn.Module):
    """A simple Deep Q-Network """

    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim[0], layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))
        self.out = nn.Linear(layers[-1], num_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.out(x)
        return x

    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)

    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def get_action(self, x, steps_done, log_print_count, PRINT_ON):
        with torch.no_grad():  # auto-grad off
            if len(x.shape) == 1:
                x = x.view(1, -1)  # 1x? 매트릭스 변환
            # kjkoo : for debugging
            fwd_x = self.forward(x)
            # kjkoo :
            if steps_done % log_print_count == 0 & PRINT_ON == True:
                print(f"[+] @get_action: fwd_x(q_vals-DQN out) :{fwd_x[0][0:5]}, max:{fwd_x.max(1)[0]}")
            return fwd_x.max(1)[1], fwd_x
            # return self.forward(x).max(1)[1]    # 행에서 max를 선택. [1] index 선택 = action number


class DQNAgent_Attacker:
    """A simple Deep Q-Network Agent """

    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=20000,
                 pre_training_steps=50000,  # kjkoo : (23-0807) 50000
                 batch_size=32,
                 replay_size=10000,
                 final_epsilon=0.05,
                 exploration_steps=10000,
                 gamma=0.99,
                 hidden_sizes=[64, 64],
                 target_update_freq=1000,
                 verbose=True,
                 **kwargs):

        # This DQN implementation only works for flat actions
        assert env.flat_actions
        self.verbose = verbose
        if self.verbose:
            print(f"\nRunning DQN with config:")
            pprint(locals())

        # set seeds
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

        # environment setup
        self.env = env

        self.num_actions = self.env.action_space.n

        self.obs_dim = self.env.A_observation_space.shape
        # logger setup
        self.logger = SummaryWriter('./runs/attacker/')

        # Training related attributes
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

        # kjkoo : (23-0807)
        self.pre_training_steps = pre_training_steps
        self.pre_steps_done = 0

        # Neural Network related attributes
        # kjkoo : (23-0810)
        #   - M1-chip, mps 적용
        USE_MPS = False
        if USE_MPS:
            if torch.cuda.is_available():
                device_type = "cuda"
            elif torch.backends.mps.is_available():
                device_type = "mps"
            else:
                device_type = "cpu"
            self.device = torch.device(device_type)
        else:
            self.device = torch.device("cuda"
                                       if torch.cuda.is_available()
                                       else "cpu")
        self.dqn = DQN(self.obs_dim,
                       hidden_sizes,
                       self.num_actions).to(self.device)
        if self.verbose:
            print(f"\nUsing Neural Network running on device={self.device}:")
            print(self.dqn)

        self.target_dqn = DQN(self.obs_dim,
                              hidden_sizes,
                              self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq

        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # replay setup
        self.replay = ReplayMemory(replay_size,
                                   self.obs_dim,
                                   self.device)

        # kjkoo : (23-0712), debug
        self.log_print_count = 200
        self.PRINT_ON = False

        # kjkoo : (23-0714), adaptive epsilon
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

        # kjkoo : (23-0714)
        #   - adaptive epsilon

    def get_epsilon_adaptive(self):
        if self.ep_pos_done < self.exploration_steps:
            return self.epsilon_schedule[self.ep_pos_done]
        return self.final_epsilon

        # kjkoo : (23-0706)
        # input : o, state, St -> DQN 입력
        # output : #action, At -> DQN 출력
        # DQN은 training이 아니라, no_grad로 inference(추론)만 진행
        # 기능 : training 초기에는 exploration(탐험)을 많이 하기 위해서,
        #   epsilon값이 1에 가까우며, random action을 많이 선택한다.
        #   하지만, 학습이 exploration step 이상 진행되면,
        #   epsilon이 final value(0.05)로 고정되며,
        #   training된 DQN의 추론에 의해 action을 선택하는 확률을 높인다.
    def get_egreedy_action(self, o, epsilon, steps):
        random_value = random.random()
        # if random.random() > epsilon:
        if random_value > epsilon:
            # kjkoo : (23-0706) for debugging
            if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
                print(
                    f"[+] @get_egreedy_action: rand#>e={epsilon} (exploitation), steps=[{steps}], o(in-state) : {o[0:15]}")
            o = torch.from_numpy(o).float().to(self.device)
            # kjkoo : (23-0712)
            # kjkoo : (23-0806)
            action_tensor, Q_val = self.dqn.get_action(o, self.steps_done, self.log_print_count, self.PRINT_ON)
            action = action_tensor.cpu().item()
            # action, Q_val = self.dqn.get_action(o, self.steps_done,
            #                                    self.log_print_count,
            #                                    self.PRINT_ON).cpu().item()

            # print(f"[+] action={action}, Q_val={Q_val}")
            return action, 'Q', random_value, epsilon
            # return self.dqn.get_action(o).cpu().item()
        else:
            if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
                print(
                    f"[+] @get_egreedy_action: rand#<e={epsilon} (exploration), steps=[{steps}], o(in-state) : {o[0:15]}")
            return random.randint(0, self.num_actions - 1), 'R', random_value, epsilon

    def optimize(self):
        # kjkoo : (23-0713)
        if self.replay.size < self.batch_size:
            return 0.0, 0.0

        batch = self.replay.sample_batch(self.batch_size, self.steps_done)
        s_batch, a_batch, next_s_batch, r_batch, d_batch = batch

        # print(f"[+] @optimize : a_batch: \n{a_batch}")

        # get q_vals for each state and the action performed in that state
        # kjkoo : (23-0805)
        #   - q_vals_raw : (batch x action_dim)
        #   - a_batch : (batch x 1)
        #   - q_vals_raw.gather(1, a_batch) : q_vals_raw에서 a 위치에 해당하는 값을 추출. batch만큼 수행
        #   - squeeze() : batch x 1 형태에서 1 차원을 없앰. q_vals vector 생성
        q_vals_raw = self.dqn(s_batch)  # 상태 s의 Q값
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()  # action(a)의 Q값

        # kjkoo
        if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
            print(f"[+] q_vals : {q_vals}")

        # get target q val = max val of next state
        # kjkoo : (23-0805)
        #   - target_q_val_raw = (batch x action_dim)
        #   - target_q_val_raw.max(1) : 각 raw(q_vals)에서 max값 선택. 반환(values, indices)
        #   - target_q_Val_raw.max(1)[0] : max반환값에서 values([0]) 선택 : batch 크기의 vector 생성. max Q값 선택
        with torch.no_grad():
            target_q_val_raw = self.target_dqn(next_s_batch)
            target_q_val = target_q_val_raw.max(1)[0]
            target = r_batch + self.discount * (1 - d_batch) * target_q_val

        if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
            print(f"[+] target : {target}")

        # calculate loss
        # kjkoo : (23-0806) ~ TD? (temporal difference)
        loss = self.loss_fn(q_vals, target)

        # kjkoo
        if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
            print(f"[+ loss : {loss.item()}")

        # optimize the model
        # kjkoo : (23-0806)
        #   - zero_grad() : 이전 step의 gradient가 누적되어 오동작하는 것을 방지하기 위해
        #   - backward() : autograd -> gradient 계산 + 파라메터 누적
        #   - step() : optimizer()에 파라메터 업데이트 위탁
        #   - [참고] https://blog.naver.com/PostView.naver?blogId=vail131&logNo=222464398222
        self.optimizer.zero_grad()
        loss.backward()
        # kjkoo : (23-0810)
        #   -
        USE_GRAD_CLAMP = False
        if USE_GRAD_CLAMP:
            for param in self.dqn.parameters():
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.steps_done % self.target_update_freq == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())

        # kjkoo : (23-0806)
        #   - q_vals_max : (batch x action_dim). 각 raw에서 max 선택. values([0]) 선택. batch크기 q_vals 벡터
        #   - q_vals_max의 평균값 계산. tensor -> 표준 python number로 반환
        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()

        # kjkoo
        if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
            print(f"[+] q_vals_max : {q_vals_max}")
            print(f"[+] mean_v (q_vals_max.mean()) : {mean_v}")

        return loss.item(), mean_v

    def pre_train(self):
        if self.verbose:
            print("\nStarting pre-training")

        num_episodes = 0
        pre_training_steps_remaining = self.pre_training_steps

        while self.pre_steps_done < self.pre_training_steps:
            # run_pre_train_episode()
            ep_results = self.run_pre_train_episode(pre_training_steps_remaining)

            ep_return, ep_steps, goal, ep_reward_list, ep_action_list, \
                ep_actres_list, ep_action_type_list, ep_ranval_list, \
                ep_epsilon_list = ep_results

            num_episodes += 1
            pre_training_steps_remaining -= ep_steps

            if True:
                # if num_episodes % 10 == 0 and self.verbose:
                print(f"\nEpisode(pre-train) {num_episodes}:")
                print(f"\tpre_steps done = {self.pre_steps_done} / "
                      f"{self.pre_training_steps}")
                print(f"\treturn = {ep_return}")
                print(f"\tgoal = {goal}")
                # kjkoo : (23-0704)
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

        # kjkoo : (23-0810)
        #   - training 진행 상황 확인
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

            ## 디버깅 용도

            # if True:
            if num_episodes < 100:
                if num_episodes % 10 == 0 and self.verbose:
                    print(f"\nEpisode {num_episodes}:")
                    print(f"\tsteps done = {self.steps_done} / "
                          f"{self.training_steps}")
                    print(f"\treturn = {ep_return}")
                    print(f"\tgoal = {goal}")
                    # kjkoo : (23-0704)
                    print(f"\tepsilon = {self.get_epsilon()}")
                    # print(f"\treward_list[{len(ep_reward_list)}] = {ep_reward_list}")
                    # print(f"\taction_list[{len(ep_action_list)}] = {ep_action_list}")
                    # print(f"\tactype_list[{len(ep_action_type_list)}] = {ep_action_type_list}")
                    # print(f"\tranvdl_list[{len(ep_ranval_list)}] = {[round(rv, 3) for rv in ep_ranval_list]}")
                    # print(f"\tepsiln_list[{len(ep_epsilon_list)}] = {[round(rv, 3) for rv in ep_epsilon_list]}")
                    # print(f"\tactres_list[{len(ep_actres_list)}] = {ep_actres_list}")
                    # print(f"\tobserv_list[{len(ep_obs_list)}] = {ep_obs_list[0], ep_obs_list[-1]}")

                    # kjkoo : (23-0809)
                    #   - nn model weight 및 gradient 출력
                    #       ; 왜? gradient explode가 있는지 확인하기 위해.
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
                    # kjkoo : (23-0704)
                    print(f"\tepsilon = {self.get_epsilon()}")
                    # print(f"\treward_list[{len(ep_reward_list)}] = {ep_reward_list}")
                    # print(f"\taction_list[{len(ep_action_list)}] = {ep_action_list}")
                    # print(f"\tactype_list[{len(ep_action_type_list)}] = {ep_action_type_list}")
                    # print(f"\tranvdl_list[{len(ep_ranval_list)}] = {[round(rv, 3) for rv in ep_ranval_list]}")
                    # print(f"\tepsiln_list[{len(ep_epsilon_list)}] = {[round(rv, 3) for rv in ep_epsilon_list]}")
                    # print(f"\tactres_list[{len(ep_actres_list)}] = {ep_actres_list}")
                    # print(f"\tobserv_list[{len(ep_obs_list)}] = {ep_obs_list[0], ep_obs_list[-1]}")

                    # kjkoo : (23-0809)
                    #   - nn model weight 및 gradient 출력
                    #       ; 왜? gradient explode가 있는지 확인하기 위해.
                    PRINT_DQN_PARAMS = False
                    if PRINT_DQN_PARAMS:
                        self.print_dqn_parameters()

                if num_episodes == 399 and self.verbose:
                    print(f"\nEpisode {num_episodes}:")
                    print(f"\tsteps done = {self.steps_done} / "
                          f"{self.training_steps}")
                    print(f"\treturn = {ep_return}")
                    print(f"\tgoal = {goal}")
                    # kjkoo : (23-0704)
                    print(f"\tepsilon = {self.get_epsilon()}")
                    # print(f"\treward_list[{len(ep_reward_list)}] = {ep_reward_list}")
                    # print(f"\taction_list[{len(ep_action_list)}] = {ep_action_list}")
                    # print(f"\tactype_list[{len(ep_action_type_list)}] = {ep_action_type_list}")
                    # print(f"\tranvdl_list[{len(ep_ranval_list)}] = {[round(rv, 3) for rv in ep_ranval_list]}")
                    # print(f"\tepsiln_list[{len(ep_epsilon_list)}] = {[round(rv, 3) for rv in ep_epsilon_list]}")
                    # print(f"\tactres_list[{len(ep_actres_list)}] = {ep_actres_list}")
                    # print(f"\tobserv_list[{len(ep_obs_list)}] = {ep_obs_list[0], ep_obs_list[-1]}")

                    # step별로 상태 및 행동 정보 확인
                    render_mode = "readable"
                    for i in range(len(ep_action_list)):
                        self.env.render(render_mode, ep_obs_list[i])
                        action = self.env.action_space.get_action(ep_action_list[i])
                        print(f"Performing[{i}]: [a:{ep_action_list[i]}] {action}")
                    self.env.render(render_mode, ep_obs_list[-1])

                    # kjkoo : (23-0809)
                    #   - nn model weight 및 gradient 출력
                    #       ; 왜? gradient explode가 있는지 확인하기 위해.
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

        # kjkoo : (23-0704) : for debugging
        reward_list = list()
        action_list = list()
        actres_list = list()
        action_type_list = list()
        ranval_list = list()
        epsilon_list = list()

        while not done and not env_step_limit_reached and steps < step_limit:
            # kjkoo : (23-0806)
            a = random.randint(0, self.num_actions-1)
            next_o, r, done, env_step_limit_reached, _ = self.env.step(a)
            # positive reward 데이터만 저장
            if r > 0:
                self.replay.store_new_data(o, a, next_o, r, done)
            self.pre_steps_done += 1

            o = next_o
            episode_return += r
            steps += 1
            # kjkoo
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

        # kjkoo : (23-0704) : for debugging
        reward_list = list()
        action_list = list()
        actres_list = list()
        action_type_list = list()
        ranval_list = list()
        epsilon_list = list()
        obs_list = list()       # observation 기록
        obs_list.append(o)      # 초기값 저장

        while not done and not env_step_limit_reached and steps < step_limit:

            # kjkoo : (23-0806)
            a, a_type, rv, es = self.get_egreedy_action(o, self.get_epsilon(), steps)
            #a = self.get_egreedy_action(o, self.get_epsilon(), steps)
            # kjkoo : (23-0714)
            #a = self.get_egreedy_action(o, self.get_epsilon_adaptive(), steps)

            # kjkoo
            if self.steps_done % self.log_print_count == 0 & self.PRINT_ON == True:
                print(f"[+] steps=[{steps}], action=[{a}]")

            next_o, r, done, env_step_limit_reached, _ = self.env.step(a)
            #self.replay.store_new_data(o, a, next_o, r, done)  # 새로운 transition 데이터만 replay memory에 저장
            #self.replay.store_new_trans(o, a, next_o, r, done)  # 새로운 transition 데이터만 replay memory에 저장
            self.replay.store(o, a, next_o, r, done)  # replay memory에 데이터 저장
            self.steps_done += 1
            # kjkoo : (23-0806) -> 의미를 모르겠음
            #if self.hold_epsilon != True:
            #    self.ep_pos_done += 1

            loss, mean_v = self.optimize()

            self.logger.add_scalar("loss", loss, self.steps_done)
            self.logger.add_scalar("mean_v(q_vals)", mean_v, self.steps_done)

            o = next_o
            episode_return += r
            steps += 1

            # kjkoo : 디버깅
            reward_list.append(r)
            action_list.append(a)
            actres_list.append(_['success'])
            action_type_list.append(a_type)
            ranval_list.append(rv)
            epsilon_list.append(es)
            obs_list.append(o)

        # kjkoo : (23-0806)
        #   - 로직 의미를 모르겠다. 필요 없으면 삭제 예정
        #if self.pre_episode_return > episode_return:
        #    self.hold_epsilon = True
        #    #self.ep_pos = self.steps_done
        #else:
        #    self.hold_epsilon = False
        #    self.pre_episode_return = episode_return


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
                        #default=[128, 128, 128],
                        help="(default=[64, 64])")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default=0.001)")
    parser.add_argument("-t", "--training_steps", type=int, default=40000,
                        help="training steps (default=50000)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="(default=32)")
    parser.add_argument("--target_update_freq", type=int, default=1000,
                        help="(default=1000)")
    parser.add_argument("--seed", type=int, default=386,     # 99
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

    # kjkoo : (23-0806) - pre_train() : 훈련 데이터를 미리 쌓는 과정
    #dqn_agent.pre_train()

    dqn_agent.train()

    # kjkoo : (23-0808)
    #   - dqn_agent policies 저장 (DQN parameters(weight)
    date = dt.datetime.now()
    file_path_date = "./etri_dqn_policies/dqn_agent" + date.strftime("-%y_%m%d_%H%M") + ".pt"
    file_path = "./etri_dqn_policies/dqn_agent.pt"
    dqn_agent.save(file_path_date)
    dqn_agent.save(file_path)

    dqn_agent.run_eval_episode(render=args.render_eval)


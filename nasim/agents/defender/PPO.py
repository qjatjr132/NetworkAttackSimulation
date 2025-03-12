import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from torch.distributions import Categorical


# Actor-Critic 네트워크 정의
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor 네트워크
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic 네트워크
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr=0.0003,
                 gamma=0.99,
                 lambda_gae=0.95,
                 entropy_coeff=0.01,
                 max_grad_norm=0.5,
                 clip_epsilon=0.2,
                 K_epochs=4):

        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.entropy_coeff = entropy_coeff

        self.max_grad_norm = max_grad_norm
        self.clip_epsilon = clip_epsilon
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.memory = []

        self.logger = SummaryWriter('./runs/PPO_defender')

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_returns(self, rewards, dones, values, next_value):
        gae=0
        returns = []

        next_value = torch.tensor([next_value], dtype=torch.float32)
        # values와 next_value를 이어붙임
        values = torch.cat((values, next_value))

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    # def compute_returns(self, rewards, dones, values, next_value):
    #     returns = []
    #     R = next_value
    #     for step in reversed(range(len(rewards))):
    #         R = rewards[step] + self.gamma * R * (1 - dones[step])  # 할인된 미래 보상 계산
    #         returns.insert(0, R)  # 미래 보상을 리스트 앞에 추가하여 최신 상태의 리턴 계산
    #     return returns

    def load_model(self, path):
        self.policy = torch.load(f'{path}/ppo.pt')

    def save_model(self, path):
        torch.save(self.policy, f'{path}/ppo.pt')

    def update(self):
        # 메모리에서 데이터를 가져옴
        state_arr, action_arr, logprob_arr, reward_arr, done_arr, value_arr = zip(*self.memory)

        # 각 변수들을 텐서로 변환
        states = torch.FloatTensor(state_arr)
        actions = torch.LongTensor(action_arr)
        old_log_probs = torch.FloatTensor(logprob_arr)
        rewards = torch.FloatTensor(reward_arr)
        dones = torch.FloatTensor(done_arr)
        values = torch.FloatTensor(value_arr)

        # Policy의 entropy 계산
        entropies = []
        for log_prob in old_log_probs:
            entropy = -torch.sum(torch.exp(log_prob) * log_prob, dim=-1)  # entropy 계산
            entropies.append(entropy)
        entropies = torch.stack(entropies)

        # 다음 상태의 값 예측
        _, next_value = self.policy(states[-1].unsqueeze(0))
        next_value = next_value.detach().item()

        # 리턴 계산
        returns = torch.FloatTensor(self.compute_returns(rewards, dones, values, next_value))

        # Advantage 계산
        advantages = returns - values

        # PPO 업데이트
        for _ in range(self.K_epochs):
            # 현재 상태에서의 행동 확률과 가치 예측
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)

            # 로그 확률의 비율 계산 (PPO의 핵심)
            ratio = torch.exp(new_logprobs - old_log_probs)

            # 클리핑된 비율과 원본 비율을 사용하여 손실 계산
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic의 손실 (값 함수 손실)
            critic_loss = self.MseLoss(state_values.squeeze(), returns)

            # entropy term을 손실에 추가
            entropy_loss = -self.entropy_coeff * entropies.mean()

            # 총 손실
            loss = actor_loss + critic_loss + entropy_loss

            # 경사 하강법 적용
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # 정책 업데이트
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 메모리 초기화
        self.memory = []
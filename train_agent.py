from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
import pickle
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import copy

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from utils import *

logger = None
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# ---------------- Actor-Critic ---------------- #
class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, gamma=0.99, hidden_sizes=[512, 256]):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.l1 = nn.Linear(state_dim, hidden_sizes[0])
        self.l2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.actor = nn.Linear(hidden_sizes[1], act_dim)
        self.critic = nn.Linear(hidden_sizes[1], 1)

        self.saved_actions = []
        self.rewards = []
        self.entropy = []

    def forward(self, inputs):
        state, act_mask = inputs
        x = F.elu(self.l1(state))
        x = F.dropout(x, p=0.5)
        x = F.elu(self.l2(x))
        x = F.dropout(x, p=0.5)

        actor_logits = self.actor(x)
        act_mask = act_mask.bool()
        actor_logits[~act_mask] = -1e9
        act_probs = F.softmax(actor_logits, dim=-1)
        state_values = self.critic(x)
        return act_probs, state_values

    def select_action(self, batch_state, batch_act_mask, device):
        state = torch.FloatTensor(batch_state).to(device)
        act_mask = torch.tensor(batch_act_mask, dtype=torch.bool).to(device)
        probs, value = self((state, act_mask))
        m = Categorical(probs)
        acts = m.sample()
        valid_idx = act_mask.gather(1, acts.view(-1, 1)).view(-1)
        acts[valid_idx == 0] = 0

        self.saved_actions.append(SavedAction(m.log_prob(acts), value))
        self.entropy.append(m.entropy())
        return acts.cpu().numpy().tolist()

    def update(self, optimizer, device, ent_weight):
        if len(self.rewards) == 0:
            del self.rewards[:]
            del self.saved_actions[:]
            del self.entropy[:]
            return 0.0, 0.0, 0.0, 0.0

        batch_rewards = np.vstack(self.rewards).T
        batch_rewards = torch.FloatTensor(batch_rewards).to(device)
        num_steps = batch_rewards.shape[1]
        for i in range(1, num_steps):
            batch_rewards[:, num_steps - i - 1] += self.gamma * batch_rewards[:, num_steps - i]

        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        for i in range(num_steps):
            log_prob, value = self.saved_actions[i]
            advantage = batch_rewards[:, i] - value.squeeze(1)
            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)
            entropy_loss += -self.entropy[i]
        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy_loss = entropy_loss.mean()
        loss = actor_loss + critic_loss + ent_weight * entropy_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del self.rewards[:]
        del self.saved_actions[:]
        del self.entropy[:]
        return loss.item(), actor_loss.item(), critic_loss.item(), entropy_loss.item()

# ---------------- DataLoader ---------------- #
class ACDataLoader(object):
    def __init__(self, uids, batch_size):
        self.uids = np.array(uids)
        self.num_users = len(uids)
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self._rand_perm = np.random.permutation(self.num_users)
        self._start_idx = 0
        self._has_next = True

    def has_next(self):
        return self._has_next

    def get_batch(self):
        if not self._has_next:
            return None
        end_idx = min(self._start_idx + self.batch_size, self.num_users)
        batch_idx = self._rand_perm[self._start_idx:end_idx]
        batch_uids = self.uids[batch_idx]
        self._has_next = end_idx < self.num_users
        self._start_idx = end_idx
        return batch_uids.tolist()

# ---------------- Training ---------------- #
def train(args):
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    uids = list(env.kg(USER).keys())
    dataloader = ACDataLoader(uids, args.batch_size)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    logger.info('Parameters:' + str([i[0] for i in model.named_parameters()]))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_losses, total_plosses, total_vlosses, total_entropy, total_rewards = [], [], [], [], []
    step = 0
    model.train()

    all_paths = {}  # {uid: [path1, path2, ...]}

    for epoch in range(1, args.epochs + 1):
        dataloader.reset()
        while dataloader.has_next():
            batch_uids = dataloader.get_batch()

            for uid in batch_uids:
                if uid not in all_paths:
                    all_paths[uid] = []

                num_samples = args.num_path_samples if hasattr(args, 'num_path_samples') else 3
                for _ in range(num_samples):
                    state = env.reset([uid])  # 单用户 reset
                    done = False
                    while not done:
                        act_mask = env.batch_action_mask(dropout=args.act_dropout)
                        act_idx = model.select_action(state, act_mask, args.device)
                        state, reward, done = env.batch_step(act_idx)
                        model.rewards.append(reward)

                    # 保存完整路径
                    all_paths[uid].append(copy.deepcopy(env._batch_path[0]))

            # ---------------- 学习更新 ---------------- #
            lr = args.lr * max(1e-4, 1.0 - float(step) / (args.epochs * len(uids) / args.batch_size))
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            total_rewards.append(np.sum(model.rewards))
            loss, ploss, vloss, eloss = model.update(optimizer, args.device, args.ent_weight)
            total_losses.append(loss)
            total_plosses.append(ploss)
            total_vlosses.append(vloss)
            total_entropy.append(eloss)
            step += 1

        # 保存模型
        policy_file = '{}/policy_model_epoch_{}.ckpt'.format(args.log_dir, epoch)
        torch.save(model.state_dict(), policy_file)
        logger.info("Saved model to " + policy_file)

        # 保存路径
        paths_file = '{}/training_paths_epoch_{}.pkl'.format(args.log_dir, epoch)
        with open(paths_file, 'wb') as f:
            pickle.dump(all_paths, f)
        logger.info(f"Saved paths to {paths_file}")

# ---------------- Main ---------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {clothing, cell, beauty, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=1, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--num_path_samples', type=int, default=3, help='Number of independent paths per user')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--ent_weight', type=float, default=1e-3, help='weight factor for entropy loss')
    parser.add_argument('--act_dropout', type=float, default=0.5, help='action dropout rate.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of hidden units')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = '{}/{}'.format(TMP_DIR[args.dataset], args.name)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    global logger
    logger = get_logger(args.log_dir + '/train_log.txt')
    logger.info(args)

    set_random_seed(args.seed)
    train(args)

if __name__ == '__main__':
    main()

from __future__ import absolute_import, division, print_function

import sys
import os
import argparse
from math import log
from datetime import datetime
from tqdm import tqdm
from collections import namedtuple
import random
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from functools import reduce

from knowledge_graph import KnowledgeGraph
from kg_env import BatchKGEnvironment
from train_agent import ActorCritic
from utils import *

# ----------------------------
# Existing evaluation utils
# ----------------------------
def evaluate(topk_matches, test_user_products):
    """Compute metrics for predicted recommendations.
    Args:
        topk_matches: a list or dict of product ids in ascending order.
    """
    invalid_users = []
    # Compute metrics
    precisions, recalls, ndcgs, hits = [], [], [], []
    test_user_idxs = list(test_user_products.keys())
    for uid in test_user_idxs:
        if uid not in topk_matches or len(topk_matches[uid]) < 10:
            invalid_users.append(uid)
            continue
        pred_list, rel_set = topk_matches[uid][::-1], test_user_products[uid]
        if len(pred_list) == 0:
            continue

        dcg = 0.0
        hit_num = 0.0
        for i in range(len(pred_list)):
            if pred_list[i] in rel_set:
                dcg += 1. / (log(i + 2) / log(2))
                hit_num += 1
        # idcg
        idcg = 0.0
        for i in range(min(len(rel_set), len(pred_list))):
            idcg += 1. / (log(i + 2) / log(2))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        recall = hit_num / len(rel_set) if len(rel_set) > 0 else 0.0
        precision = hit_num / len(pred_list) if len(pred_list) > 0 else 0.0
        hit = 1.0 if hit_num > 0.0 else 0.0

        ndcgs.append(ndcg)
        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)

    avg_precision = np.mean(precisions) * 100 if len(precisions) > 0 else 0.0
    avg_recall = np.mean(recalls) * 100 if len(recalls) > 0 else 0.0
    avg_ndcg = np.mean(ndcgs) * 100 if len(ndcgs) > 0 else 0.0
    avg_hit = np.mean(hits) * 100 if len(hits) > 0 else 0.0
    print('NDCG={:.3f} |  Recall={:.3f} | HR={:.3f} | Precision={:.3f} | Invalid users={}'.format(
            avg_ndcg, avg_recall, avg_hit, avg_precision, len(invalid_users)))


# ----------------------------
# Beam search used for test predictions (unchanged)
# ----------------------------
def batch_beam_search(env, model, uids, device, topk=[25, 5, 1]):
    def _batch_acts_to_masks(batch_acts):
        batch_masks = []
        for acts in batch_acts:
            num_acts = len(acts)
            act_mask = np.zeros(model.act_dim, dtype=np.uint8)
            act_mask[:num_acts] = 1
            batch_masks.append(act_mask)
        return np.vstack(batch_masks)

    state_pool = env.reset(uids)  # numpy of [bs, dim]
    path_pool = env._batch_path  # list of list, size=bs
    probs_pool = [[] for _ in uids]
    model.eval()
    with torch.no_grad():
        for hop in range(3):
            state_tensor = torch.FloatTensor(state_pool).to(device)
            acts_pool = env._batch_get_actions(path_pool, False)  # list of list, size=bs
            actmask_pool = _batch_acts_to_masks(acts_pool)  # numpy of [bs, dim]
            actmask_tensor = torch.ByteTensor(actmask_pool).to(device)
            probs, _ = model((state_tensor, actmask_tensor))  # Tensor of [bs, act_dim]
            probs = probs + actmask_tensor.float()  # In order to differ from masked actions
            topk_probs, topk_idxs = torch.topk(probs, topk[hop], dim=1)  # LongTensor of [bs, k]
            topk_idxs = topk_idxs.detach().cpu().numpy()
            topk_probs = topk_probs.detach().cpu().numpy()

            new_path_pool, new_probs_pool = [], []
            for row in range(topk_idxs.shape[0]):
                path = path_pool[row]
                probs = probs_pool[row]
                for idx, p in zip(topk_idxs[row], topk_probs[row]):
                    if idx >= len(acts_pool[row]):  # act idx is invalid
                        continue
                    relation, next_node_id = acts_pool[row][idx]  # (relation, next_node_id)
                    if relation == SELF_LOOP:
                        next_node_type = path[-1][1]
                    else:
                        next_node_type = KG_RELATION[path[-1][1]][relation]
                    new_path = path + [(relation, next_node_type, next_node_id)]
                    new_path_pool.append(new_path)
                    new_probs_pool.append(probs + [p])
            path_pool = new_path_pool
            probs_pool = new_probs_pool
            if hop < 2:
                state_pool = env._batch_get_state(path_pool)

    return path_pool, probs_pool


def predict_paths(policy_file, path_file, args):
    print('Predicting paths (beam search) ...')
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = torch.load(policy_file, map_location=lambda s, l: s)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)

    test_labels = load_labels(args.dataset, 'test')
    test_uids = list(test_labels.keys())

    batch_size = 16
    start_idx = 0
    all_paths, all_probs = [], []
    pbar = tqdm(total=len(test_uids))
    while start_idx < len(test_uids):
        end_idx = min(start_idx + batch_size, len(test_uids))
        batch_uids = test_uids[start_idx:end_idx]
        paths, probs = batch_beam_search(env, model, batch_uids, args.device, topk=args.topk)
        all_paths.extend(paths)
        all_probs.extend(probs)
        start_idx = end_idx
        pbar.update(batch_size)
    predicts = {'paths': all_paths, 'probs': all_probs}
    pickle.dump(predicts, open(path_file, 'wb'))
    print(f"Saved predicted test paths to {path_file}")


# ----------------------------
# New: sample train-user paths (stochastic sampling using learned policy)
# ----------------------------
def sample_train_paths_for_faithfulness(policy_file, out_path, args, num_users=50, num_paths=1000):
    """
    Using the trained policy, randomly sample `num_users` users from the TRAIN set;
    for each user sample `num_paths` complete paths (stochastic actions via policy.select_action).
    Save result as a dict {uid: [path1, path2, ...]} to out_path (pickle).
    """
    print("Sampling training paths for faithfulness (this may take a while)...")
    # Prepare environment and model
    env = BatchKGEnvironment(args.dataset, args.max_acts, max_path_len=args.max_path_len, state_history=args.state_history)
    pretrain_sd = torch.load(policy_file, map_location=lambda s, l: s)
    model = ActorCritic(env.state_dim, env.act_dim, gamma=args.gamma, hidden_sizes=args.hidden).to(args.device)
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model.eval()

    # Load train user list
    train_labels = load_labels(args.dataset, 'train')
    train_uids = list(train_labels.keys())
    if len(train_uids) == 0:
        raise RuntimeError("Train user list empty.")

    sampled_uids = random.sample(train_uids, min(num_users, len(train_uids)))
    print(f"Selected {len(sampled_uids)} train users to sample.")

    all_paths = {}  # uid -> list of paths
    # We'll use no_grad to avoid any autograd overhead and clear saved_actions periodically.
    with torch.no_grad():
        for uid in tqdm(sampled_uids):
            all_paths[uid] = []
            # To reduce memory growth in ActorCritic (it stores saved_actions etc.), clear them before user sampling
            model.saved_actions = []
            model.rewards = []
            model.entropy = []
            # Sample paths
            for _ in range(num_paths):
                # reset environment for single user u
                state = env.reset([uid])  # numpy array [1, state_dim]
                done = False
                # run episode until done
                while not done:
                    # get action mask for batch of size 1
                    act_mask = env.batch_action_mask(dropout=0.0)  # shape [1, act_dim]
                    # model.select_action expects lists, so pass [state[0]] and [act_mask[0]]
                    # It will append saved_actions — we clear them per user so this is okay.
                    action_list = model.select_action([state[0]], [act_mask[0]], args.device)
                    # action_list is a list (len=1), need to wrap as list for env.batch_step
                    state, reward, done = env.batch_step([action_list[0]])
                # After episode finished, env._batch_path[0] holds this sampled path
                # Deepcopy to avoid pointers to mutable objects
                all_paths[uid].append(copy.deepcopy(env._batch_path[0]))
            # clear again
            model.saved_actions = []
            model.rewards = []
            model.entropy = []

    # Save to file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(all_paths, f)
    print(f"Saved sampled training paths to {out_path}")


# ----------------------------
# Evaluate predicted paths -> produce top-k product labels and run recommendation metrics
# ----------------------------
def evaluate_paths(path_file, train_labels, test_labels, args):
    embeds = load_embed(args.dataset)
    user_embeds = embeds[USER]
    purchase_embeds = embeds[PURCHASE][0]
    product_embeds = embeds[PRODUCT]
    scores = np.dot(user_embeds + purchase_embeds, product_embeds.T)

    # 1) Get all valid paths for each user, compute path score and path probability.
    results = pickle.load(open(path_file, 'rb'))
    pred_paths = {uid: {} for uid in test_labels}
    for path, probs in zip(results['paths'], results['probs']):
        if path[-1][1] != PRODUCT:
            continue
        uid = path[0][2]
        if uid not in pred_paths:
            continue
        pid = path[-1][2]
        if pid not in pred_paths[uid]:
            pred_paths[uid][pid] = []
        path_score = scores[uid][pid]
        path_prob = reduce(lambda x, y: x * y, probs)
        pred_paths[uid][pid].append((path_score, path_prob, path))

    # 2) Pick best path for each user-product pair, also remove pid if it is in train set.
    best_pred_paths = {}
    for uid in pred_paths:
        train_pids = set(train_labels[uid])
        best_pred_paths[uid] = []
        for pid in pred_paths[uid]:
            if pid in train_pids:
                continue
            # Get the path with highest probability
            sorted_path = sorted(pred_paths[uid][pid], key=lambda x: x[1], reverse=True)
            best_pred_paths[uid].append(sorted_path[0])

    # 3) Compute top 10 recommended products for each user.
    sort_by = 'score'
    pred_labels = {}
    for uid in best_pred_paths:
        if sort_by == 'score':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[0], x[1]), reverse=True)
        elif sort_by == 'prob':
            sorted_path = sorted(best_pred_paths[uid], key=lambda x: (x[1], x[0]), reverse=True)
        top10_pids = [p[-1][2] for _, _, p in sorted_path[:10]]  # from largest to smallest
        # add up to 10 pids if not enough
        if args.add_products and len(top10_pids) < 10:
            train_pids = set(train_labels[uid])
            cand_pids = np.argsort(scores[uid])
            for cand_pid in cand_pids[::-1]:
                if cand_pid in train_pids or cand_pid in top10_pids:
                    continue
                top10_pids.append(cand_pid)
                if len(top10_pids) >= 10:
                    break
        # end of add
        pred_labels[uid] = top10_pids[::-1]  # change order to from smallest to largest!

    evaluate(pred_labels, test_labels)


# ----------------------------
# Main test() orchestrator
# ----------------------------
def test(args):
    policy_file = os.path.join(args.log_dir, 'policy_model_epoch_{}.ckpt'.format(args.epochs))
    path_file = os.path.join(args.log_dir, 'policy_paths_epoch{}.pkl'.format(args.epochs))
    train_sampled_file = os.path.join(args.log_dir, 'train_sampled_paths.pkl')

    train_labels = load_labels(args.dataset, 'train')
    test_labels = load_labels(args.dataset, 'test')

    # 1) If requested, sample training paths for faithfulness (50 users x 1000 paths default)
    if args.sample_train_faith:
        # sample and save train paths
        sample_train_paths_for_faithfulness(policy_file, train_sampled_file, args,
                                           num_users=args.num_train_users,
                                           num_paths=args.num_paths_per_user)

    # 2) Generate test paths (beam search)
    if args.run_path:
        predict_paths(policy_file, path_file, args)

    # 3) Evaluate (recommendation metrics) if requested
    if args.run_eval:
        evaluate_paths(path_file, train_labels, test_labels, args)


# ----------------------------
# CLI & entrypoint
# ----------------------------
if __name__ == '__main__':
    boolean = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    parser.add_argument('--seed', type=int, default=123, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', type=int, default=1, help='num of epochs.')
    parser.add_argument('--max_acts', type=int, default=250, help='Max number of actions.')
    parser.add_argument('--max_path_len', type=int, default=3, help='Max path length.')
    parser.add_argument('--gamma', type=float, default=0.99, help='reward discount factor.')
    parser.add_argument('--state_history', type=int, default=1, help='state history length')
    parser.add_argument('--hidden', type=int, nargs='*', default=[512, 256], help='number of samples')
    parser.add_argument('--add_products', type=boolean, default=False, help='Add predicted products up to 10')
    parser.add_argument('--topk', type=int, nargs='*', default=[25, 5, 1], help='beam sizes')
    parser.add_argument('--run_path', type=boolean, default=True, help='Generate predicted path? (takes long time)')
    parser.add_argument('--run_eval', type=boolean, default=True, help='Run evaluation?')
    # New args for faithfulness sampling
    parser.add_argument('--sample_train_faith', type=boolean, default=True,
                        help='If true, sample train users paths for faithfulness (50 users x 1000 paths by default)')
    parser.add_argument('--num_train_users', type=int, default=50, help='Number of train users to sample')
    parser.add_argument('--num_paths_per_user', type=int, default=1000, help='Number of paths per train user to sample')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    test(args)

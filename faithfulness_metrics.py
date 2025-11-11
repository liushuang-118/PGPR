import pickle
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
import random
import os

# ==============================
# 配置文件路径
# ==============================
dataset_dir = 'tmp/Amazon_Beauty/train_agent'
training_pkl = os.path.join(dataset_dir, 'training_paths_epoch_1.pkl')
policy_pkl = os.path.join(dataset_dir, 'policy_paths_epoch1.pkl')
F_u_pkl = os.path.join(dataset_dir, 'F_u.pkl')

num_samples_per_user = 100  # 每个用户采样路径数

# ==============================
# 函数：路径转规则
# ==============================
def path_to_rule(path):
    # path: [(relation, node_type, node_id), ...]
    return tuple([step[0] for step in path])  # 只保留关系序列

# ==============================
# 读取训练路径并计算 F(u)
# ==============================
with open(training_pkl, 'rb') as f:
    training_paths = pickle.load(f)

print(f"Loaded training paths for {len(training_paths)} users.")

# # 打印基本信息
# print(f"Number of users: {len(training_paths)}")
# first_user = list(training_paths.keys())[0]
# print(f"First user ID: {first_user}")
# print(f"Number of paths for this user: {len(training_paths[first_user])}")

# # 打印该用户的前 3 条路径
# print("\nExample paths for the first user:")
# for i, path in enumerate(training_paths[first_user][:3]):
#     print(f"Path {i+1}: {path}")

# # 打印前 5 用户的前 1 条路径
# print("\nFirst path for first 5 users:")
# for uid in list(training_paths.keys())[:5]:
#     print(f"User {uid}: {training_paths[uid][0]}")

F_u_dict = dict()
for u, paths in training_paths.items():
    if len(paths) > num_samples_per_user:
        paths = random.sample(paths, num_samples_per_user)
    rules = [path_to_rule(p) for p in paths]
    count = Counter(rules)
    total = sum(count.values())
    F_u = {rule: freq / total for rule, freq in count.items()}
    F_u_dict[u] = F_u

sample_users = random.sample(list(F_u_dict.keys()), 5)

for u in sample_users:
    print(f"User {u}:")
    F_u = F_u_dict[u]
    # 按概率从大到小排序
    sorted_rules = sorted(F_u.items(), key=lambda x: x[1], reverse=True)
    for rule, prob in sorted_rules[:5]:  # 只打印概率最高的 5 条规则
        print(f"  Rule {rule} -> {prob:.4f}")
    print()

# # 保存 F(u)
# with open(F_u_pkl, 'wb') as f:
#     pickle.dump(F_u_dict, f)
# print(f"Saved F(u) to {F_u_pkl}")

# # ==============================
# # 读取策略生成路径
# # ==============================
# with open(policy_pkl, 'rb') as f:
#     policy_paths_data = pickle.load(f)

# # policy_paths_data 可能是字典，每个用户对应 dict 或 list
# policy_paths = dict()
# for u, info in policy_paths_data.items():
#     if isinstance(info, dict) and 'paths' in info:
#         policy_paths[u] = info['paths']
#     else:
#         policy_paths[u] = info

# print(f"Loaded policy paths for {len(policy_paths)} users.")

# # ==============================
# # 计算 Jensen-Shannon divergence
# # ==============================
# def compute_JS(F_dict, policy_dict, weighted=False):
#     """
#     F_dict: {user_id: {rule: prob, ...}, ...}
#     policy_dict: {user_id: [path1, path2, ...], ...}
#     weighted: 是否按规则概率加权
#     """
#     js_list = []

#     for u, F_u in F_dict.items():
#         if u not in policy_dict:
#             continue
#         paths = policy_dict[u]
#         rules = [path_to_rule(p) for p in paths]
#         count = Counter(rules)
#         total = sum(count.values())
#         Q_u = {rule: freq / total for rule, freq in count.items()}

#         # 统一规则集合
#         all_rules = set(F_u.keys()) | set(Q_u.keys())
#         p = np.array([F_u.get(r, 0.0) for r in all_rules])
#         q = np.array([Q_u.get(r, 0.0) for r in all_rules])

#         # Jensen-Shannon divergence
#         js = jensenshannon(p, q, base=2.0)
#         if weighted and len(F_u) > 0:
#             # 可按 F_u 权重加权
#             js *= np.sum(list(F_u.values()))
#         js_list.append(js)

#     if len(js_list) == 0:
#         return 0.0
#     return np.mean(js_list)

# # ==============================
# # 计算 JSf 和 JSw
# # ==============================
# JSf = compute_JS(F_u_dict, policy_paths, weighted=False)
# JSw = compute_JS(F_u_dict, policy_paths, weighted=True)

# print(f"JSf = {JSf:.6f}")
# print(f"JSw = {JSw:.6f}")

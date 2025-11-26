# import pickle
# from collections import Counter, defaultdict
# import numpy as np
# # ========= 参数 =========
# DATASET = "Amazon_Beauty"

# # ========= 自动生成路径 =========
# BASE_DIR = f'tmp/{DATASET}/train_agent'

# F_u_file = f'{BASE_DIR}/training_paths_epoch_1.pkl'
# test_file = f'{BASE_DIR}/policy_paths_epoch1.pkl'

# # ========= 工具函数 =========
# def extract_relations(path):
#     """提取关系模式，忽略 self_loop"""
#     return tuple(step[0] for step in path if isinstance(step, tuple) and step[0] != "self_loop")

# # ========= 读取训练集 F(u) =========
# with open(F_u_file, 'rb') as f:
#     train_data = pickle.load(f)
# F_u = {}
# for user_id, paths in train_data.items():
#     cnt = Counter()
#     for path in paths:
#         rels = extract_relations(path)
#         if rels:
#             cnt[rels] += 1
#     total = sum(cnt.values())
#     F_u[user_id] = {r: c / total for r, c in cnt.items()} if total > 0 else {}
# print(f"训练用户数量: {len(F_u)}")

# # ========= 读取测试集 =========
# with open(test_file, 'rb') as f:
#     test_data = pickle.load(f)

# test_paths = test_data['paths']  # 所有路径
# test_probs = test_data.get('probs', [])  # 对应的概率列表

# print(f"测试路径总数: {len(test_paths)}")

# # 整合每个用户的路径和对应概率
# user_paths = defaultdict(list)  # key: user_id, value: list of (path, prob)
# for idx, path in enumerate(test_paths):
#     # 找到路径中的 user 节点
#     user_id = None
#     for step in path:
#         if step[0] == 'self_loop' and step[1] == 'user':
#             user_id = step[2]
#             break
#     if user_id is None:
#         continue
#     prob = test_probs[idx] if idx < len(test_probs) else 1.0  # 如果没有 probs 默认1.0
#     user_paths[user_id].append((path, prob))

# print(f"整合后的测试用户数量: {len(user_paths)}")

# # 构建每个用户的规则分布，并只保留出现频率前10的规则，同时保留对应概率
# user_rule_distributions = {}
# for user_id, paths_with_probs in user_paths.items():
#     cnt = Counter()
#     path_to_prob = {}  # 保存每个路径模式对应的概率
#     for path, prob in paths_with_probs:
#         rule = extract_relations(path)
#         if rule:
#             cnt[rule] += 1
#             # 如果一个路径模式出现多次，可以累加或者取平均概率
#             if rule in path_to_prob:
#                 path_to_prob[rule].append(prob)
#             else:
#                 path_to_prob[rule] = [prob]
#     total = sum(cnt.values())
#     # 排序取前10
#     top_rules = cnt.most_common(10)
#     dist = {}
#     probs = {}
#     for r, c in top_rules:
#         dist[r] = c / total  # 出现概率
#         # 对应概率取平均
#         probs[r] = np.mean(path_to_prob[r])
#     user_rule_distributions[user_id] = (dist, probs)

# # # 打印前3个用户示例
# # for i, (uid, (dist, probs)) in enumerate(user_rule_distributions.items()):
# #     print(f"\n用户 {uid} 的规则分布 (共 {len(user_paths[uid])} 条路径, 只保留前10条模式):")
# #     for rule in dist:
# #         print(f"  {rule}: 出现概率={dist[rule]:.3f}, 平均路径概率={probs[rule]:.3f}")
# #     if i >= 2:
# #         break

# # ========= JS 函数 =========
# EPS = 1e-12

# def kl(p, q):
#     p = np.array(p) + EPS
#     q = np.array(q) + EPS
#     return np.sum(p * np.log(p / q))

# def js_divergence(p_dict, q_dict):
#     """计算两个离散分布的 JS divergence"""
#     keys = set(p_dict.keys()) | set(q_dict.keys())
#     p = np.array([p_dict.get(k, 0.0) for k in keys])
#     q = np.array([q_dict.get(k, 0.0) for k in keys])
#     p /= (p.sum() + EPS)
#     q /= (q.sum() + EPS)
#     m = 0.5 * (p + q)
#     return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# # ========= 计算 JSf / JSw =========
# js_f_list = []
# js_w_list = []
# skipped = 0

# for user_id, (Qf, Qw) in user_rule_distributions.items():
#     # F(u) 是训练阶段的规则分布
#     Fu = F_u.get(user_id, {})
#     if not Fu or not Qf:
#         skipped += 1
#         continue

#     # JSf: 直接用出现频率
#     js_f = js_divergence(Qf, Fu)
#     js_f_list.append(js_f)

#     # JSw: 用路径概率加权
#     # 将 Qw 转换为归一化分布
#     sum_w = sum(Qw.values()) + EPS
#     Qw_norm = {k: v / sum_w for k, v in Qw.items()}
#     js_w = js_divergence(Qw_norm, Fu)
#     js_w_list.append(js_w)

# # 求平均
# JSf = np.mean(js_f_list) if js_f_list else float('nan')
# JSw = np.mean(js_w_list) if js_w_list else float('nan')

# print("\n========= Faithfulness 指标 =========")
# print(f"JSf = {JSf:.6f}")
# print(f"JSw = {JSw:.6f}")
# print(f"(跳过 {skipped} 个用户)")

import pickle
from collections import Counter, defaultdict
import numpy as np

# ========= 参数 =========
DATASET = "Amazon_Beauty"
BASE_DIR = f'tmp/{DATASET}/train_agent'

# ========= 文件路径 =========
F_u_file = f'{BASE_DIR}/train_sampled_paths.pkl'      # 训练采样路径 (50 个用户)
test_file = f'{BASE_DIR}/policy_paths_epoch1.pkl'     # 测试用户生成路径

TOPK_PRODUCTS = 20  # 每个用户取 top-k 推荐产品对应路径

# ========= 工具函数 =========
def extract_relations(path):
    """提取路径关系模式，忽略 self_loop"""
    return tuple(step[0] for step in path if isinstance(step, tuple) and step[0] != "self_loop")

def kl(p, q, eps=1e-12):
    p = np.array(p) + eps
    q = np.array(q) + eps
    return np.sum(p * np.log(p / q))

def js_divergence(p_dict, q_dict, eps=1e-12):
    """计算两个离散分布的 JS divergence"""
    keys = set(p_dict.keys()) | set(q_dict.keys())
    p = np.array([p_dict.get(k, 0.0) for k in keys])
    q = np.array([q_dict.get(k, 0.0) for k in keys])
    p /= (p.sum() + eps)
    q /= (q.sum() + eps)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m, eps) + 0.5 * kl(q, m, eps)

# ========= 读取训练采样路径 F(u) =========
with open(F_u_file, 'rb') as f:
    train_data = pickle.load(f)

F_u = {}
for user_id, paths in train_data.items():
    cnt = Counter()
    for path in paths:
        rels = extract_relations(path)
        if rels:
            cnt[rels] += 1
    total = sum(cnt.values())
    F_u[user_id] = {r: c / total for r, c in cnt.items()} if total > 0 else {}

print(f"训练采样用户数量: {len(F_u)}")

# ========= 读取测试集 policy_paths =========
with open(test_file, 'rb') as f:
    test_data = pickle.load(f)

test_paths = test_data['paths']
test_probs = test_data.get('probs', [])

# 整合每个用户的路径和对应概率
user_paths = defaultdict(list)  # key: user_id, value: list of (path, prob)
for idx, path in enumerate(test_paths):
    user_id = None
    for step in path:
        if step[0] == 'self_loop' and step[1] == 'user':
            user_id = step[2]
            break
    if user_id is None or user_id not in F_u:  # 只考虑抽样的 50 个用户
        continue
    prob = test_probs[idx] if idx < len(test_probs) else 1.0
    user_paths[user_id].append((path, prob))

print(f"测试采样用户数量: {len(user_paths)}")

# ========= 为每个用户选 Top-K 产品路径 =========
user_top_paths = {}
for user_id, paths_with_probs in user_paths.items():
    # 假设每条路径最后一步是 PRODUCT
    product_to_paths = defaultdict(list)
    for path, prob in paths_with_probs:
        pid = path[-1][2]
        product_to_paths[pid].append((path, prob))
    # 按路径概率求和，取 top-k 产品
    product_scores = {pid: sum(p for _, p in plist) for pid, plist in product_to_paths.items()}
    top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:TOPK_PRODUCTS]

    # 取每个 top 产品的所有路径
    top_paths = []
    for pid, _ in top_products:
        top_paths.extend(product_to_paths[pid])
    user_top_paths[user_id] = top_paths

# ========= 构建规则分布 =========
user_rule_distributions = {}
for user_id, paths_with_probs in user_top_paths.items():
    cnt = Counter()
    path_to_prob = defaultdict(list)
    for path, prob in paths_with_probs:
        rule = extract_relations(path)
        if rule:
            cnt[rule] += 1
            path_to_prob[rule].append(prob)
    total = sum(cnt.values())
    dist = {r: c/total for r, c in cnt.items()} if total > 0 else {}
    probs = {r: np.mean(path_to_prob[r]) for r in path_to_prob} if path_to_prob else {}
    user_rule_distributions[user_id] = (dist, probs)

# ========= 计算 JSf / JSw =========
js_f_list = []
js_w_list = []
skipped = 0

for user_id, (Qf, Qw) in user_rule_distributions.items():
    Fu = F_u.get(user_id, {})
    if not Fu or not Qf:
        skipped += 1
        continue

    # JSf: 直接用出现频率
    js_f_list.append(js_divergence(Qf, Fu))

    # JSw: 用路径概率加权
    sum_w = sum(Qw.values())
    if sum_w == 0:
        skipped += 1
        continue
    Qw_norm = {k: v/sum_w for k, v in Qw.items()}
    js_w_list.append(js_divergence(Qw_norm, Fu))

JSf = np.mean(js_f_list) if js_f_list else float('nan')
JSw = np.mean(js_w_list) if js_w_list else float('nan')

print("\n========= Faithfulness 指标 =========")
print(f"JSf = {JSf:.6f}")
print(f"JSw = {JSw:.6f}")
print(f"(跳过 {skipped} 个用户)")

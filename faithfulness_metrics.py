import pickle
from collections import Counter, defaultdict
import numpy as np
# ========= 参数 =========
DATASET = "Amazon_Beauty"

# ========= 自动生成路径 =========
BASE_DIR = f'tmp/{DATASET}/train_agent'

F_u_file = f'{BASE_DIR}/training_paths_epoch_1.pkl'
test_file = f'{BASE_DIR}/policy_paths_epoch1.pkl'

# ========= 工具函数 =========
def extract_relations(path):
    """提取关系模式，忽略 self_loop"""
    return tuple(step[0] for step in path if isinstance(step, tuple) and step[0] != "self_loop")

# ========= 读取训练集 F(u) =========
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
print(f"训练用户数量: {len(F_u)}")

# ========= 读取测试集 =========
with open(test_file, 'rb') as f:
    test_data = pickle.load(f)

test_paths = test_data['paths']  # 所有路径
test_probs = test_data.get('probs', [])  # 对应的概率列表

print(f"测试路径总数: {len(test_paths)}")

# 整合每个用户的路径和对应概率
user_paths = defaultdict(list)  # key: user_id, value: list of (path, prob)
for idx, path in enumerate(test_paths):
    # 找到路径中的 user 节点
    user_id = None
    for step in path:
        if step[0] == 'self_loop' and step[1] == 'user':
            user_id = step[2]
            break
    if user_id is None:
        continue
    prob = test_probs[idx] if idx < len(test_probs) else 1.0  # 如果没有 probs 默认1.0
    user_paths[user_id].append((path, prob))

print(f"整合后的测试用户数量: {len(user_paths)}")

# 构建每个用户的规则分布，并只保留出现频率前10的规则，同时保留对应概率
user_rule_distributions = {}
for user_id, paths_with_probs in user_paths.items():
    cnt = Counter()
    path_to_prob = {}  # 保存每个路径模式对应的概率
    for path, prob in paths_with_probs:
        rule = extract_relations(path)
        if rule:
            cnt[rule] += 1
            # 如果一个路径模式出现多次，可以累加或者取平均概率
            if rule in path_to_prob:
                path_to_prob[rule].append(prob)
            else:
                path_to_prob[rule] = [prob]
    total = sum(cnt.values())
    # 排序取前10
    top_rules = cnt.most_common(10)
    dist = {}
    probs = {}
    for r, c in top_rules:
        dist[r] = c / total  # 出现概率
        # 对应概率取平均
        probs[r] = np.mean(path_to_prob[r])
    user_rule_distributions[user_id] = (dist, probs)

# # 打印前3个用户示例
# for i, (uid, (dist, probs)) in enumerate(user_rule_distributions.items()):
#     print(f"\n用户 {uid} 的规则分布 (共 {len(user_paths[uid])} 条路径, 只保留前10条模式):")
#     for rule in dist:
#         print(f"  {rule}: 出现概率={dist[rule]:.3f}, 平均路径概率={probs[rule]:.3f}")
#     if i >= 2:
#         break

# ========= JS 函数 =========
EPS = 1e-12

def kl(p, q):
    p = np.array(p) + EPS
    q = np.array(q) + EPS
    return np.sum(p * np.log(p / q))

def js_divergence(p_dict, q_dict):
    """计算两个离散分布的 JS divergence"""
    keys = set(p_dict.keys()) | set(q_dict.keys())
    p = np.array([p_dict.get(k, 0.0) for k in keys])
    q = np.array([q_dict.get(k, 0.0) for k in keys])
    p /= (p.sum() + EPS)
    q /= (q.sum() + EPS)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# ========= 计算 JSf / JSw =========
js_f_list = []
js_w_list = []
skipped = 0

for user_id, (Qf, Qw) in user_rule_distributions.items():
    # F(u) 是训练阶段的规则分布
    Fu = F_u.get(user_id, {})
    if not Fu or not Qf:
        skipped += 1
        continue

    # JSf: 直接用出现频率
    js_f = js_divergence(Qf, Fu)
    js_f_list.append(js_f)

    # JSw: 用路径概率加权
    # 将 Qw 转换为归一化分布
    sum_w = sum(Qw.values()) + EPS
    Qw_norm = {k: v / sum_w for k, v in Qw.items()}
    js_w = js_divergence(Qw_norm, Fu)
    js_w_list.append(js_w)

# 求平均
JSf = np.mean(js_f_list) if js_f_list else float('nan')
JSw = np.mean(js_w_list) if js_w_list else float('nan')

print("\n========= Faithfulness 指标 =========")
print(f"JSf = {JSf:.6f}")
print(f"JSw = {JSw:.6f}")
print(f"(跳过 {skipped} 个用户)")
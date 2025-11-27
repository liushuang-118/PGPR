import pickle
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import jensenshannon

# ========= 参数 =========
DATASET = "Amazon_Beauty"
BASE_DIR = f"tmp/{DATASET}/train_agent"
TRAIN_FILE = f"{BASE_DIR}/train_sampled_paths.pkl"
TEST_FILE  = f"{BASE_DIR}/policy_paths_epoch50.pkl"
TOPK_PRODUCTS = 20

# ========= 有效关系 =========
VALID_RELATIONS = [
    "purchase", "produced_by", "belongs_to", "also_bought",
    "also_viewed", "bought_together", "mentions", "described_as"
]

# ================================
# 1. 构建训练 F(u)
# ================================
with open(TRAIN_FILE, "rb") as f:
    train_data = pickle.load(f)

F_u = defaultdict(Counter)

for path in train_data['paths']:
    uid = None
    for rel, typ, idx in path:
        if typ == "user" and uid is None:
            uid = idx
        if rel in VALID_RELATIONS:
            F_u[uid][rel] += 1

# 归一化
F_u_norm = {}
for uid, counter in F_u.items():
    total = sum(counter.values())
    if total > 0:
        F_u_norm[uid] = {rel: count / total for rel, count in counter.items()}

print(f"[INFO] Built F(u) for {len(F_u_norm)} users")

# ========= 2. 处理测试路径 (仅训练用户) =========
with open(TEST_FILE, "rb") as f:
    test_data = pickle.load(f)

paths = test_data['paths']
probs = test_data['probs']

train_users = set(F_u_norm.keys())

user_prod_scores = defaultdict(lambda: defaultdict(float))
user_prod_paths  = defaultdict(lambda: defaultdict(list))

for path, p_list in zip(paths, probs):
    path_prob = float(np.prod(p_list))
    uid, pid = None, None
    for rel, typ, idx in path:
        if typ == "user" and uid is None:
            uid = idx
        elif typ == "product":
            pid = idx
    if uid is None or pid is None:
        continue
    if uid not in train_users:  # 只保留训练用户
        continue

    # 累积每个产品概率
    user_prod_scores[uid][pid] += path_prob
    # 保存所有路径及其概率
    user_prod_paths[uid][pid].append({
        "path": path,
        "prob": path_prob
    })

# 选 top-K 产品及每个产品最大概率路径
user_topk_paths = defaultdict(list)
for uid in train_users:
    score_dict = user_prod_scores.get(uid, {})
    if not score_dict:
        continue
    sorted_prods = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    topk_pids = [pid for pid, _ in sorted_prods[:TOPK_PRODUCTS]]
    for pid in topk_pids:
        sorted_paths = sorted(user_prod_paths[uid][pid], key=lambda x: x['prob'], reverse=True)
        if sorted_paths:
            user_topk_paths[uid].append(sorted_paths[0])  # 取概率最高的路径

print(f"[INFO] Got top-{TOPK_PRODUCTS} products & paths for {len(user_topk_paths)} users")

# ================================
# 3. 构建测试规则分布 Qf(u) / Qw(u)
# ================================
Qf_u = defaultdict(Counter)
Qw_u = defaultdict(Counter)

for uid, path_list in user_topk_paths.items():
    for item in path_list:
        path = item['path']
        prob = item['prob']  # 可作为权重计算 Qw(u)
        for rel, typ, idx in path:
            if rel in VALID_RELATIONS:
                Qf_u[uid][rel] += 1          # 计数
                Qw_u[uid][rel] += prob       # 权重累积

# 归一化 Qf(u)
Qf_u_norm = {}
for uid, counter in Qf_u.items():
    total = sum(counter.values())
    if total > 0:
        Qf_u_norm[uid] = {rel: count / total for rel, count in counter.items()}

# 归一化 Qw(u)
Qw_u_norm = {}
for uid, counter in Qw_u.items():
    total = sum(counter.values())
    if total > 0:
        Qw_u_norm[uid] = {rel: val / total for rel, val in counter.items()}

print("[INFO] Built Qf(u) and Qw(u) distributions")

# ================================
# 4. 计算 JSf / JSw
# ================================
jsf_list, jsw_list = [], []

for uid in train_users:
    if uid not in Qf_u_norm or uid not in F_u_norm:
        continue

    rels = set(F_u_norm[uid].keys()) | set(Qf_u_norm[uid].keys())
    P = np.array([F_u_norm[uid].get(r, 0) for r in rels])
    Qf = np.array([Qf_u_norm[uid].get(r, 0) for r in rels])
    Qw = np.array([Qw_u_norm[uid].get(r, 0) for r in rels])

    jsf = jensenshannon(P, Qf)**2
    jsw = jensenshannon(P, Qw)**2
    jsf_list.append(jsf)
    jsw_list.append(jsw)

JSf = np.mean(jsf_list)
JSw = np.mean(jsw_list)

print(f"JSf = {JSf:.6f}")
print(f"JSw = {JSw:.6f}")


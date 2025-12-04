import pickle
from collections import defaultdict, Counter
import numpy as np
from scipy.spatial.distance import jensenshannon
import random

# ========= 参数 =========
DATASET = "Amazon_Beauty"
BASE_DIR = f"tmp/{DATASET}/train_agent"
TRAIN_FILE = f"{BASE_DIR}/train_sampled_paths.pkl"
TEST_FILE  = f"{BASE_DIR}/matched_policy_paths.pkl"

TOPK_PRODUCTS = 20
NUM_PATHS_PER_PRODUCT = 50

VALID_RELATIONS = [
    "purchase", "produced_by", "belongs_to", "also_bought",
    "also_viewed", "bought_together", "mentions", "described_as"
]

# ================================
# 1. 构建训练分布 F(u)
# ================================
with open(TRAIN_FILE, "rb") as f:
    train_data = pickle.load(f)

F_u = defaultdict(Counter)

for path in train_data["paths"]:
    uid = None
    for rel, typ, idx in path:
        if typ == "user" and uid is None:
            uid = idx
        if rel in VALID_RELATIONS:
            F_u[uid][rel] += 1

# normalize
F_u_norm = {}
for uid, cnt in F_u.items():
    total = sum(cnt.values())
    if total > 0:
        F_u_norm[uid] = {rel: c / total for rel, c in cnt.items()}

train_users = set(F_u_norm.keys())
print(f"[INFO] Built F(u) for {len(train_users)} train users")

# ================================
# 2. 读取测试文件（兼容所有格式）
# ================================
with open(TEST_FILE, "rb") as f:
    raw = pickle.load(f)

# 自动推断结构
if isinstance(raw, dict) and "paths" in raw:
    paths = raw["paths"]
    probs = raw["probs"]

elif isinstance(raw, list):
    # case1: [paths, probs]
    if len(raw) == 2 and isinstance(raw[0], list):
        paths, probs = raw

    # case2: [(path, prob), ...]
    elif len(raw) > 0 and isinstance(raw[0], tuple) and len(raw[0]) == 2:
        paths = [x[0] for x in raw]
        probs = [x[1] for x in raw]

    else:
        raise ValueError(f"[ERROR] Unknown list format in file: {TEST_FILE}")

else:
    raise ValueError(f"[ERROR] Unsupported file format: {type(raw)}")

print(f"[INFO] Loaded test paths = {len(paths)}")

# ================================
# 3. 构建用户–商品 路径概率累加
# ================================
user_prod_scores = defaultdict(lambda: defaultdict(float))
user_prod_paths  = defaultdict(lambda: defaultdict(list))

test_users = set()

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

    test_users.add(uid)

    if uid not in train_users:
        continue

    user_prod_scores[uid][pid] += path_prob
    user_prod_paths[uid][pid].append({"path": path, "prob": path_prob})

print(f"[INFO] Found {len(test_users)} test users")
print(f"[INFO] Intersection with train users = {len(train_users & test_users)} users")

# 🌟 overlapping users
target_users = train_users & test_users
if len(target_users) == 0:
    print("[ERROR] No overlapping users between train and test!!!")
    exit()

print(f"[INFO] Evaluating JS divergence for {len(target_users)} users")

# ================================
# 4. 构建 Qf(u), Qw(u)
# ================================
user_topk_paths = defaultdict(list)

for uid in target_users:
    score_dict = user_prod_scores.get(uid, {})
    if not score_dict:
        continue

    # ---- (1) 自适应 TOP-K 商品 ----
    # 如果商品不足 20 个，则只取实际数量
    sorted_prods = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    topk_pids = [pid for pid, _ in sorted_prods[:min(TOPK_PRODUCTS, len(sorted_prods))]]

    # ---- (2) 每个商品自适应抽样 ----
    for pid in topk_pids:
        p_list = user_prod_paths[uid][pid]

        # 若路径不足 NUM_PATHS_PER_PRODUCT，则全部使用
        k = min(NUM_PATHS_PER_PRODUCT, len(p_list))

        # 路径为空时跳过
        if k == 0:
            continue

        sampled_paths = random.sample(p_list, k)
        user_topk_paths[uid].extend(sampled_paths)

print(f"[INFO] Got sampled paths for {len(user_topk_paths)} users")

# 构建 Qf(u) / Qw(u)
Qf_u = defaultdict(Counter)
Qw_u = defaultdict(Counter)

for uid, p_list in user_topk_paths.items():
    for item in p_list:
        path = item["path"]
        prob = item["prob"]
        for rel, typ, idx in path:
            if rel in VALID_RELATIONS:
                Qf_u[uid][rel] += 1
                Qw_u[uid][rel] += prob

# 归一化
Qf_u_norm = {}
Qw_u_norm = {}

for uid, cnt in Qf_u.items():
    tot = sum(cnt.values())
    if tot > 0:
        Qf_u_norm[uid] = {rel: c / tot for rel, c in cnt.items()}

for uid, cnt in Qw_u.items():
    tot = sum(cnt.values())
    if tot > 0:
        Qw_u_norm[uid] = {rel: v / tot for rel, v in cnt.items()}

print("[INFO] Built Qf(u) and Qw(u)")

# ================================
# 5. 计算 JSf / JSw
# ================================
jsf_list, jsw_list = [], []

for uid in target_users:
    if uid not in Qf_u_norm or uid not in F_u_norm:
        continue

    rels = set(F_u_norm[uid].keys()) | set(Qf_u_norm[uid].keys())
    P   = np.array([F_u_norm[uid].get(r, 0) for r in rels])
    Qf  = np.array([Qf_u_norm[uid].get(r, 0) for r in rels])
    Qw  = np.array([Qw_u_norm[uid].get(r, 0) for r in rels])

    jsf_list.append(jensenshannon(P, Qf) ** 2)
    jsw_list.append(jensenshannon(P, Qw) ** 2)

JSf = np.mean(jsf_list)
JSw = np.mean(jsw_list)

print("\n====== JS Divergence Results ======")
print(f"JSf = {JSf:.6f}")
print(f"JSw = {JSw:.6f}")

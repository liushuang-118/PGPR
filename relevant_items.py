import os
import pickle
import gzip
from collections import defaultdict

# ================================
# 配置路径
# ================================
DATASET = "Amazon_Beauty"
POLICY_FILE = f"tmp/{DATASET}/train_agent/policy_paths_epoch50.pkl"
TEST_FILE   = f"data/{DATASET}/test.txt.gz"
OUTPUT_FILE = f"tmp/{DATASET}/train_agent/matched_policy_paths.pkl"


# ================================
# 1. 加载真实购买记录
# ================================
purchased = defaultdict(set)  # uid -> set of purchased pids

with gzip.open(TEST_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        # 每行格式: uid pid [word ids]
        parts = line.strip().split()
        uid = int(parts[0])
        pid = int(parts[1])
        purchased[uid].add(pid)

print(f"[INFO] Loaded purchased products for {len(purchased)} users")

# ================================
# 2. 加载 policy paths
# ================================
with open(POLICY_FILE, "rb") as f:
    data = pickle.load(f)

paths = data["paths"]
probs = data["probs"]

print(f"[INFO] Loaded {len(paths)} paths")

# ================================
# 3. 提取匹配真实购买的路径
# ================================
matched_paths = []  # [(path, prob)]

for path, prob in zip(paths, probs):
    uid = None
    for rel, typ, idx in path:
        if typ == "user":
            uid = idx
            break
    if uid is None or uid not in purchased:
        continue

    last_rel, last_typ, last_pid = path[-1]  # 路径最后一个 tuple
    if last_typ == "product" and last_pid in purchased[uid]:
        matched_paths.append((path, prob))  # 只保存 path 和 prob

print(f"[INFO] Found {len(matched_paths)} paths corresponding to real purchases")

# ================================
# 4. 保存 matched_paths
# ================================
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump(matched_paths, f)

print(f"[INFO] Saved matched paths to {OUTPUT_FILE}")
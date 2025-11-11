import os
import gzip
import pickle
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import AmazonDataset
from itertools import islice

# ===== 配置路径 =====
DATA_DIR = './data/Amazon_Beauty'
PATH_FILE = './tmp/Amazon_Beauty/train_agent/policy_paths_epoch1.pkl'

# ===== 加载 reasoning paths =====
with open(PATH_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"[INFO] 已加载 {len(data['paths'])} 条 reasoning paths")

# ===== 统计每种实体类型的流行度 =====
entity_popularity_by_type = defaultdict(lambda: defaultdict(int))
for path in data['paths']:
    for rel, etype, eid in path:
        if etype != 'word':
            entity_popularity_by_type[etype][eid] += 1

# ===== 对每种实体类型做 min-max 归一化 =====
entity_popularity_normalized = {}
for etype, pop_dict in entity_popularity_by_type.items():
    min_val = min(pop_dict.values())
    max_val = max(pop_dict.values())
    if max_val == min_val:
        # 防止除以0
        entity_popularity_normalized[etype] = {eid: 0.0 for eid in pop_dict}
    else:
        entity_popularity_normalized[etype] = {eid: (v - min_val) / (max_val - min_val)
                                               for eid, v in pop_dict.items()}


user_products_tmp = defaultdict(list)   # {uid: [(pid, path_prob), ...]}
user_topk_paths_tmp = defaultdict(list) # {uid: [(path, path_prob), ...]}

for path, probs in zip(data['paths'], data['probs']):
    user_id = None
    last_product_id = None
    path_prob = np.prod(probs)  # 用路径中每步概率的乘积近似路径整体概率

    for rel, ent_type, ent_id in path:
        if ent_type == 'user' and user_id is None:
            user_id = ent_id
        elif ent_type == 'product':
            last_product_id = ent_id

    # 确保路径有效
    if user_id is None or last_product_id is None:
        continue

    # 收集候选产品和路径
    user_products_tmp[user_id].append((last_product_id, path_prob))
    user_topk_paths_tmp[user_id].append((path, path_prob))

# ===== 仅保留每个用户前 10 个推荐产品与前 K 条路径 =====
TOP_P = 10
K = 10
user_products = {}
user_topk_paths = {}

for uid, items in user_products_tmp.items():
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
    user_products[uid] = [pid for pid, _ in sorted_items[:TOP_P]]

for uid, paths in user_topk_paths_tmp.items():
    sorted_paths = sorted(paths, key=lambda x: x[1], reverse=True)
    user_topk_paths[uid] = [p for p, _ in sorted_paths[:K]]

print(f"[INFO] 提取完成: {len(user_products)} 个用户")
print(f"[INFO] 每个用户 Top-{TOP_P} 推荐产品与 Top-{K} 路径已生成")


# ===== 计算 SEP =====
beta_sep = 0.5
user_sep = {}

for uid, paths in user_topk_paths.items():
    sep_scores = []

    for path in paths:
        # 提取所有非 word 实体的归一化流行度值
        sep_values = [entity_popularity_normalized[etype].get(eid, 0.0)
                      for _, etype, eid in path if etype != 'word']

        if not sep_values:
            continue

        # 初始化：SEP(e1, v1) = v1
        sep_score = sep_values[0]

        # 指数衰减递推：SEP(e_i, v_i) = (1 - β) * SEP(e_{i-1}, v_{i-1}) + β * v_i
        for v in sep_values[1:]:
            sep_score = (1 - beta_sep) * sep_score + beta_sep * v

        sep_scores.append(sep_score)

    # 用户 SEP 为其 top-k 路径的平均 SEP
    if sep_scores:
        user_sep[uid] = np.mean(sep_scores)

# ===== 4️⃣ 计算 ETD（修改：路径类型取最后一个关系） =====
# 先统计全局所有路径的最后关系数量
global_last_rels = set()
for path in data['paths']:
    last_rel = None
    for rel, etype, eid in path:
        if etype != 'word':
            last_rel = rel
    if last_rel is not None:
        global_last_rels.add(last_rel)
total_global_last_rels = len(global_last_rels)

user_etd = {}
for uid, paths in user_topk_paths.items():
    last_rels = set()
    for path in paths:
        last_rel = None
        for rel, etype, eid in path:
            if etype != 'word':
                last_rel = rel
        if last_rel is not None:
            last_rels.add(last_rel)
    etd_score = len(last_rels) / min(K, total_global_last_rels)
    user_etd[uid] = etd_score


# ===== 5️⃣ 输出所有用户平均 SEP 和 ETD =====
avg_sep = np.mean(list(user_sep.values()))
avg_etd = np.mean(list(user_etd.values()))

print(f"平均 SEP: {avg_sep:.4f}")
print(f"平均 ETD: {avg_etd:.4f}")

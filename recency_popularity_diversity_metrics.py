import os
import gzip
import pickle
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import AmazonDataset
from itertools import islice
from knowledge_graph import KnowledgeGraph
import pandas as pd

# ===== 配置路径 =====
DATA_DIR = './data/Amazon_Beauty'
PATH_FILE = './tmp/Amazon_Beauty/train_agent/policy_paths_epoch1.pkl'

# ===== 加载 reasoning paths =====
with open(PATH_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"[INFO] 已加载 {len(data['paths'])} 条 reasoning paths")

# ===== 重新使用 KG 的全局度数作为流行度 =====
dataset = AmazonDataset(DATA_DIR)
KG = KnowledgeGraph(dataset)
KG.compute_degrees()

# ===== 统计每种实体类型的流行度 =====
entity_popularity_by_type = defaultdict(lambda: defaultdict(int))
for path in data['paths']:
    for rel, etype, eid in path:
        if etype != 'word':
            entity_popularity_by_type[etype][eid] += 1

# ===== 对每种实体类型做 min-max 归一化 =====
entity_popularity_normalized = {}

for etype, deg_dict in KG.degrees.items():
    values = list(deg_dict.values())
    min_val, max_val = min(values), max(values)

    if max_val == min_val:
        # 避免除0
        entity_popularity_normalized[etype] = {eid: 0.0 for eid in deg_dict}
    else:
        entity_popularity_normalized[etype] = {
            eid: (deg - min_val) / (max_val - min_val)
            for eid, deg in deg_dict.items()
        }

print("[INFO] 全局实体流行度加载完成（来自 KG 度数）")


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
        # 提取每个实体的全局 popularity（已归一化）
        sep_values = []
        for rel, etype, eid in path:
            # 忽略 word 实体
            if etype == 'word':
                continue

            # 从全局 popularity 中查询
            pop = entity_popularity_normalized.get(etype, {}).get(eid, 0.0)
            sep_values.append(pop)

        if not sep_values:
            continue

        # 初始化：SEP(e1) = v1
        sep_score = sep_values[0]

        # 指数衰减公式
        for v in sep_values[1:]:
            sep_score = (1 - beta_sep) * sep_score + beta_sep * v

        sep_scores.append(sep_score)

    if sep_scores:
        user_sep[uid] = np.mean(sep_scores)

# ===== 计算 ETD（修改：路径类型取最后一个关系） =====
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


# ===== 输出所有用户平均 SEP 和 ETD =====
avg_sep = np.mean(list(user_sep.values()))
avg_etd = np.mean(list(user_etd.values()))

print(f"平均 SEP: {avg_sep:.4f}")
print(f"平均 ETD: {avg_etd:.4f}")


# ===== 配置 =====
TIME_TRAIN_FILE = './data/Amazon_Beauty/time_train.csv'
TIME_TEST_FILE = './data/Amazon_Beauty/time_test.csv'
PATH_FILE = './tmp/Amazon_Beauty/train_agent/policy_paths_epoch1.pkl'
TOP_K_PATHS = 10
BETA_LIR = 0.5  # temporal decay factor

# ===== 加载训练集和测试集 =====
train_df = pd.read_csv(TIME_TRAIN_FILE)
test_df = pd.read_csv(TIME_TEST_FILE)
all_time_df = pd.concat([train_df, test_df], ignore_index=True)

# ===== 转换日期为 datetime =====
all_time_df['PURCHASE_Time'] = pd.to_datetime(all_time_df['PURCHASE_Time'], format='%Y-%m-%d', errors='coerce')
all_time_df = all_time_df.dropna(subset=['PURCHASE_Time'])

# ===== 转换为天数，从最早日期算起 =====
min_date = all_time_df['PURCHASE_Time'].min()
all_time_df['PURCHASE_Time_days'] = (all_time_df['PURCHASE_Time'] - min_date).dt.days

# ===== 构建用户-商品时间字典 {uid: {pid: days}} =====
user_item_time = defaultdict(dict)
for row in all_time_df.itertuples(index=False):
    uid, pid, t_days = row.UID, row.PID, row.PURCHASE_Time_days
    user_item_time[uid][pid] = t_days

# ===== 计算每条路径的 LIR（未归一化） =====
user_lir_raw = {}  # 存每个用户前 K 条路径的 LIR列表
for uid, paths in user_topk_paths.items():
    lir_scores = []
    for path in paths[:TOP_K_PATHS]:
        # 提取路径中所有产品实体的时间
        product_times = []
        for rel, etype, eid in path:
            if etype in ['product', 'related_product']:
                ts = user_item_time.get(uid, {}).get(eid, None)
                if ts is not None:
                    product_times.append(ts)
        if not product_times:
            continue

        # 按时间顺序排序
        product_times.sort()

        # 递归计算 EWMA LIR
        lir = product_times[0]
        for t in product_times[1:]:
            lir = (1 - BETA_LIR) * lir + BETA_LIR * t

        lir_scores.append(lir)

    if lir_scores:
        user_lir_raw[uid] = lir_scores

# ===== 对每个用户的 LIR 做 min-max 归一化 =====
user_lir = {}
for uid, lir_list in user_lir_raw.items():
    min_lir = min(lir_list)
    max_lir = max(lir_list)
    if max_lir == min_lir:
        user_lir[uid] = 0.0
    else:
        normalized = [(x - min_lir) / (max_lir - min_lir) for x in lir_list]
        user_lir[uid] = np.mean(normalized)

# ===== 全体用户平均 LIR =====
avg_lir = np.mean(list(user_lir.values()))
print(f"[INFO] 平均 LIR (归一化到 [0,1]): {avg_lir:.4f}")
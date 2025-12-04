import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_utils import AmazonDataset
from knowledge_graph import KnowledgeGraph

# ===== 配置 =====
DATASET_NAME = 'Amazon_Beauty'
MATCHED_FILE = f'./tmp/{DATASET_NAME}/train_agent/matched_policy_paths.pkl'
TIME_TRAIN_FILE = f'./data/{DATASET_NAME}/time_train.csv'
TIME_TEST_FILE = f'./data/{DATASET_NAME}/time_test.csv'

BETA_LIR = 0.5
BETA_SEP = 0.5
TOP_K_PATHS = 10

# ===== 加载 matched paths =====
with open(MATCHED_FILE, 'rb') as f:
    matched_paths_data = pickle.load(f)

# matched_paths_data 格式: [(path, prob), ...]
user_topk_paths_tmp = defaultdict(list)
for path, prob in matched_paths_data:
    user_id = None
    for rel, etype, eid in path:
        if etype == 'user':
            user_id = eid
            break
    if user_id is not None:
        user_topk_paths_tmp[user_id].append(path)  # 只保留 path

# ===== 使用 KG 全局度数作为实体流行度 =====
DATA_DIR = f'./data/{DATASET_NAME}'
dataset = AmazonDataset(DATA_DIR)
KG = KnowledgeGraph(dataset)
KG.compute_degrees()

entity_popularity_normalized = {}
for etype, deg_dict in KG.degrees.items():
    values = list(deg_dict.values())
    min_val, max_val = min(values), max(values)
    if max_val == min_val:
        entity_popularity_normalized[etype] = {eid: 0.0 for eid in deg_dict}
    else:
        entity_popularity_normalized[etype] = {
            eid: (deg - min_val) / (max_val - min_val) for eid, deg in deg_dict.items()
        }

# ===== SEP =====
user_sep = {}
for uid, paths in user_topk_paths_tmp.items():
    if not paths:
        continue
    sep_scores = []
    for path in paths[:TOP_K_PATHS]:  # 实际路径数量少于 TOP_K_PATHS 也没关系
        sep_values = [
            entity_popularity_normalized.get(etype, {}).get(eid, 0.0)
            for rel, etype, eid in path if etype != 'word'
        ]
        if not sep_values:
            continue
        sep_score = sep_values[0]
        for v in sep_values[1:]:
            sep_score = (1 - BETA_SEP) * sep_score + BETA_SEP * v
        sep_scores.append(sep_score)
    if sep_scores:
        user_sep[uid] = np.mean(sep_scores)
avg_sep = np.mean(list(user_sep.values()))
print(f"[INFO] 平均 SEP: {avg_sep:.4f}")

# ===== ETD =====
global_last_rels = set()
for path, _ in matched_paths_data:
    last_rel = None
    for rel, etype, eid in path:
        if etype != 'word':
            last_rel = rel
    if last_rel:
        global_last_rels.add(last_rel)
total_global_last_rels = len(global_last_rels)

user_etd = {}
for uid, paths in user_topk_paths_tmp.items():
    if not paths:
        continue
    last_rels = set()
    for path in paths[:TOP_K_PATHS]:
        last_rel = None
        for rel, etype, eid in path:
            if etype != 'word':
                last_rel = rel
        if last_rel:
            last_rels.add(last_rel)
    # 用实际路径数量计算比例
    user_etd[uid] = len(last_rels) / min(len(paths[:TOP_K_PATHS]), total_global_last_rels)
avg_etd = np.mean(list(user_etd.values()))
print(f"[INFO] 平均 ETD: {avg_etd:.4f}")

# ===== LIR =====
train_df = pd.read_csv(TIME_TRAIN_FILE)
test_df = pd.read_csv(TIME_TEST_FILE)
all_time_df = pd.concat([train_df, test_df], ignore_index=True)
all_time_df['PURCHASE_Time'] = pd.to_datetime(all_time_df['PURCHASE_Time'], errors='coerce')
all_time_df = all_time_df.dropna(subset=['PURCHASE_Time'])
min_date = all_time_df['PURCHASE_Time'].min()
all_time_df['PURCHASE_Time_days'] = (all_time_df['PURCHASE_Time'] - min_date).dt.days

user_item_time = defaultdict(dict)
for row in all_time_df.itertuples(index=False):
    user_item_time[row.UID][row.PID] = row.PURCHASE_Time_days
    
user_lir_raw = {}
for uid, paths in user_topk_paths_tmp.items():
    if not paths:
        continue
    lir_scores = []
    for path in paths[:TOP_K_PATHS]:
        product_times = [
            user_item_time.get(uid, {}).get(eid)
            for rel, etype, eid in path if etype in ['product', 'related_product']
        ]
        product_times = [t for t in product_times if t is not None]
        if not product_times:
            continue
        product_times.sort()
        lir = product_times[0]
        for t in product_times[1:]:
            lir = (1 - BETA_LIR) * lir + BETA_LIR * t
        lir_scores.append(lir)
    if lir_scores:
        user_lir_raw[uid] = lir_scores

user_lir = {}
for uid, lir_list in user_lir_raw.items():
    min_lir = min(lir_list)
    max_lir = max(lir_list)
    if max_lir == min_lir:
        user_lir[uid] = 0.0
    else:
        user_lir[uid] = np.mean([(x - min_lir) / (max_lir - min_lir) for x in lir_list])
avg_lir = np.mean(list(user_lir.values()))
print(f"[INFO] 平均 LIR: {avg_lir:.4f}")

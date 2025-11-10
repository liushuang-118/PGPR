import os
import gzip
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from tqdm import tqdm
from itertools import islice
from data_utils import AmazonDataset

# ===== 配置路径 =====
DATA_DIR = './data/Amazon_Beauty'  # 数据集目录
REVIEW_FILE = os.path.join(DATA_DIR, 'train.txt.gz')  # 评论文件
PATH_FILE = './tmp/Amazon_Beauty/train_agent/policy_paths_epoch1.pkl'  # reasoning paths

# ===== 1️⃣ 加载 reasoning paths =====
with open(PATH_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"[INFO] 已加载 {len(data['paths'])} 条 reasoning paths")

# 提取所有用户ID（path的第一个节点）
all_user_ids = [path[0][2] for path in data['paths']]

# 不同用户数量
unique_user_ids = set(all_user_ids)
print(f"reasoning paths 中共有 {len(unique_user_ids)} 个不同用户")

# ===== 2️⃣ 加载用户映射文件 =====
USER_FILE = os.path.join(DATA_DIR, 'users.txt.gz')
user2id, id2user = {}, {}
with gzip.open(USER_FILE, 'rt') as f:
    for i, line in enumerate(f):
        uid = line.strip()
        user2id[uid] = i
        id2user[i] = uid
print(f"[INFO] 成功加载 {len(user2id)} 个用户 ID 映射")

# # ===== 3️⃣ 提取每个用户的 Su（路径中的 word） =====
user_explanations = defaultdict(set)
for path in data['paths']:
    # 第一个节点是用户
    user_int_id = path[0][2]
    # 直接用 int ID 作为 key，不映射到 id2user，避免丢失
    for rel, ent_type, ent_id in path:
        if ent_type == 'word':
            user_explanations[user_int_id].add(ent_id)  # 保留整数

# # ===== 4️⃣ 从 train.txt.gz 构建每个用户的评论文本 =====
dataset = AmazonDataset(DATA_DIR, set_name='train')
user_groundtruth = defaultdict(set)  # Gu
for user_idx, product_idx, word_indices in dataset.review.data:
    user_groundtruth[user_idx].update(word_indices)

print(f"[INFO] 构建了 {len(user_groundtruth)} 个用户的真实评论词集合")

# 5️⃣ 构建用户评论文本列表（词 ID 转词）
user_texts = {}
for user_idx, word_indices in user_groundtruth.items():
    # dataset.word.vocab 是索引 -> 词表
    words = [dataset.word.vocab[wid] for wid in word_indices]
    user_texts[user_idx] = words

# 5️⃣ 计算 TF-IDF 并取 top-k 词作为 Gu
top_k = 10
vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
texts = [v for v in user_texts.values()]
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = np.array(vectorizer.get_feature_names_out())

user_gu = {}  # Gu
for i, (user_idx, _) in enumerate(user_texts.items()):
    row = tfidf_matrix[i].toarray()[0]
    top_idx = np.argsort(row)[-top_k:]  # 取 top_k
    user_gu[user_idx] = set(feature_names[top_idx])

print(f"[INFO] 已生成 {len(user_gu)} 个用户的 Gu（TF-IDF top-{top_k}）")

# ===== 6️⃣ 计算 Precision / Recall / F1 =====
precisions = []
recalls = []
f1s = []

for uid in user_explanations:
    Su = user_explanations[uid]
    Gu = user_groundtruth.get(uid, set())
    if not Gu or not Su:
        continue  # 忽略没有数据的用户

    inter = Su & Gu
    precision = len(inter) / (len(Su) + 1)  # +1 避免除零
    recall = len(inter) / (len(Gu) + 1)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 避免除零

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

print(f"平均 Precision: {np.mean(precisions):.4f}")
print(f"平均 Recall:    {np.mean(recalls):.4f}")
print(f"平均 F1:        {np.mean(f1s):.4f}")

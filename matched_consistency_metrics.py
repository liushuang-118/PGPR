import os
import gzip
import pickle
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================
# 配置
# ============================
DATASET = "Amazon_Beauty"
DATA_DIR = f"./data/{DATASET}"
VOCAB_FILE = f"{DATA_DIR}/vocab.txt.gz"
TEST_FILE = f"{DATA_DIR}/test.txt.gz"
TRAIN_FILE = f"{DATA_DIR}/train.txt.gz"
MATCHED_FILE = f"./tmp/{DATASET}/train_agent/matched_policy_paths.pkl"

# ============================
# 1. 加载 vocab
# ============================
vocab = []
with gzip.open(VOCAB_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        vocab.append(line.strip())

print(f"[INFO] Loaded vocab with {len(vocab)} words")


# ============================
# 2. 加载 Su（系统解释词）
# ============================
with open(MATCHED_FILE, "rb") as f:
    matched_paths_data = pickle.load(f)

user_Su = defaultdict(set)

for path, _ in matched_paths_data:
    uid = None

    for rel, etype, eid in path:
        if etype == "user":
            uid = eid
            break

    if uid is None:
        continue

    for rel, etype, eid in path:
        if etype == "word":
            user_Su[uid].add(eid)

print(f"[INFO] Collected Su for {len(user_Su)} users")

example_users = list(user_Su.keys())[:3]
print("\n=== Example Su (first 3 users) ===")
for u in example_users:
    print(f"User {u}: word_ids={list(user_Su[u])[:10]}")


# ============================
# 3. 从 test 集读取 Gu（真实评论词）
# ============================
user_Gu = defaultdict(set)

with gzip.open(TEST_FILE, "rt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        uid = int(parts[0])
        words = parts[2].strip("[]").split(",")
        for w in words:
            w = w.strip()
            if w != "":
                user_Gu[uid].add(int(w))

print(f"[INFO] Built Gu for {len(user_Gu)} users")

print("\n=== Example Gu (first 3 users) ===")
for u in list(user_Gu.keys())[:3]:
    print(f"User {u}: word_ids={list(user_Gu[u])[:10]}")


# # ============================
# # 4. 使用训练集构建 TF–IDF
# # ============================
# train_texts = []

# with gzip.open(TRAIN_FILE, "rt", encoding="utf-8") as f:
#     for line in f:
#         parts = line.strip().split()
#         if len(parts) < 3:
#             continue
#         word_ids = parts[2].strip("[]").split(",")
#         words = [vocab[int(w)] for w in word_ids if w.strip() != ""]
#         if words:
#             train_texts.append(" ".join(words))

# print(f"[INFO] Training documents for TF-IDF: {len(train_texts)}")

# tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=2)
# X = tfidf.fit_transform(train_texts)

# names = tfidf.get_feature_names_out()
# avg_tfidf = np.asarray(X.mean(axis=0)).ravel()
# word2tfidf = dict(zip(names, avg_tfidf))

# print(f"[INFO] TF-IDF vocabulary size: {len(word2tfidf)}")


# # ============================
# # 5. 高频 + 低 TF-IDF 词过滤
# # ============================
# all_words = [w for ws in user_Gu.values() for w in ws]
# freq = Counter(all_words)

# remove_words = set()

# for wid, f in freq.items():
#     if f > 5000:
#         word = vocab[wid]
#         if word2tfidf.get(word, 1.0) < 0.1:
#             remove_words.add(wid)

# print(f"[INFO] Removing {len(remove_words)} high-frequency & low-TFIDF words")

# # 过滤 Gu
# for uid in user_Gu:
#     user_Gu[uid] = {w for w in user_Gu[uid] if w not in remove_words}


# ============================
# 6. 计算 Precision / Recall / F1
# ============================
pre_list, rec_list, f1_list = [], [], []

for uid in tqdm(user_Su.keys(), desc="Evaluating"):
    Su = user_Su[uid]
    Gu = user_Gu.get(uid, set())

    if not Su or not Gu:
        continue

    inter = Su & Gu

    precision = len(inter) / (len(Su) + 1)
    recall = len(inter) / (len(Gu) + 1)
    f1 = 2 * precision * recall / (precision + recall + 1)

    pre_list.append(precision)
    rec_list.append(recall)
    f1_list.append(f1)

print("\n===== Explainability Evaluation =====")
print(f"Precision: {np.mean(pre_list):.4f}")
print(f"Recall:    {np.mean(rec_list):.4f}")
print(f"F1:        {np.mean(f1_list):.4f}")

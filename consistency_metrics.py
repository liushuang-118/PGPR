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

# ===== 1️⃣ 加载 reasoning paths =====
with open(PATH_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"[INFO] 已加载 {len(data['paths'])} 条 reasoning paths")

# ===== 2️⃣ 提取每个用户的 Su 和推荐产品集合 =====
user_explanations = defaultdict(set)  # Su
user_products = defaultdict(list)     # 推荐产品集合，保留顺序和概率

for path, probs in zip(data['paths'], data['probs']):
    user_id = None
    last_product_id = None
    words = set()
    path_prob = np.prod(probs)  # 可以用路径概率的乘积或平均作为该路径的综合概率

    for rel, ent_type, ent_id in path:
        if ent_type == 'user' and user_id is None:
            user_id = ent_id
        elif ent_type == 'product':
            last_product_id = ent_id
        elif ent_type == 'word':
            words.add(ent_id)

    if user_id is not None:
        user_explanations[user_id].update(words)
        if last_product_id is not None:
            user_products[user_id].append((last_product_id, path_prob))

# 对每个用户按概率排序，取 top-10 产品
for uid in user_products:
    sorted_products = sorted(user_products[uid], key=lambda x: x[1], reverse=True)
    top_products = [pid for pid, _ in sorted_products[:10]]
    user_products[uid] = top_products

# ===== 3️⃣ 加载真实评论数据 =====
dataset = AmazonDataset(DATA_DIR, set_name='train')
print(f"[INFO] 数据集中共有 {len(dataset.review.data)} 条评论")

# ===== 4️⃣ 构建 Gu（用户对推荐产品的真实评论词集合） =====
user_groundtruth = defaultdict(set)

for user_idx, product_idx, word_indices in dataset.review.data:
    if product_idx in user_products[user_idx]:
        user_groundtruth[user_idx].update(word_indices)

print(f"[INFO] 构建了 {len(user_groundtruth)} 个用户的 Gu（初始真实评论词集合）")

# ===== 5️⃣ 统计全局词频 =====
all_word_indices = [wid for words in user_groundtruth.values() for wid in words]
word_freq = Counter(all_word_indices)

# ===== 6️⃣ 计算 TF-IDF =====
user_texts = []
user_ids = []
for uid, word_indices in user_groundtruth.items():
    words = [dataset.word.vocab[w] for w in word_indices]
    user_texts.append(" ".join(words))
    user_ids.append(uid)

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_matrix = vectorizer.fit_transform(user_texts)
feature_names = np.array(vectorizer.get_feature_names_out())
avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
word2tfidf = dict(zip(feature_names, avg_tfidf))

# ===== 7️⃣ 过滤高频低TF-IDF词 =====
filtered_words = set()
for wid, freq in word_freq.items():
    if freq > 5000:
        word = dataset.word.vocab[wid]
        tfidf_val = word2tfidf.get(word, 0)
        if tfidf_val < 0.1:
            filtered_words.add(wid)

# ===== 8️⃣ 更新 Gu =====
for uid in user_groundtruth:
    user_groundtruth[uid] = {w for w in user_groundtruth[uid] if w not in filtered_words}

print(f"[INFO] 过滤高频低TF-IDF词后，Gu 更新完成")

# ===== 9️⃣ 计算 Precision / Recall / F1 =====
precisions, recalls, f1s = [], [], []
printed = 0 

for uid in tqdm(user_explanations.keys(), desc="Evaluating"):
    Su = user_explanations[uid]
    Gu = user_groundtruth.get(uid, set())

    if not Su or not Gu:
        continue

    inter = Su & Gu
    precision = len(inter) / (len(Su) + 1)
    recall = len(inter) / (len(Gu) + 1)
    f1 = 2 * precision * recall / (precision + recall + 1)

    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)

    if printed < 2:
        su_words = [dataset.word.vocab[w] for w in Su]
        gu_words = [dataset.word.vocab[w] for w in Gu]
        inter_words = [dataset.word.vocab[w] for w in inter]
        print(f"\n用户 {uid}:")
        print(f"  Su (预测词): {su_words[:10]}... 总数 {len(Su)}")
        print(f"  Gu (真实词): {gu_words[:10]}... 总数 {len(Gu)}")
        print(f"  交集: {inter_words[:10]}... 总数 {len(inter)}")
        printed += 1

if precisions:
    print("\n===== Evaluation Results =====")
    print(f"平均 Precision: {np.mean(precisions):.4f}")
    print(f"平均 Recall:    {np.mean(recalls):.4f}")
    print(f"平均 F1:        {np.mean(f1s):.4f}")
else:
    print("[WARN] 没有匹配的用户用于计算。")

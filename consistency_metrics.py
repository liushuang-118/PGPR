import pickle
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import re
from tqdm import tqdm

# ======================================================
# 1. 加载 PGPR 的输出路径文件
# ======================================================
path_file = './tmp/Amazon_Beauty/train_agent/policy_paths_epoch1.pkl'

with open(path_file, 'rb') as f:
    data = pickle.load(f)

paths = data['paths']
print(f"[INFO] 已加载 {len(paths)} 条 reasoning paths")

# ======================================================
# 2. 加载 Amazon Reviews 数据
# ======================================================
print("[INFO] 加载 Amazon Reviews-2023 (Beauty)...")
ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
reviews = ds["full"]  # full split

print("[INFO] 示例样本:")
print(reviews[0])

# ======================================================
# 3. 构建用户 -> 评论文本 字典
# ======================================================
user_reviews = defaultdict(list)
for r in tqdm(reviews, desc="Processing reviews"):
    if "user_id" in r and r["user_id"] and isinstance(r["text"], str):
        user_reviews[r["user_id"]].append(r["text"])

print(f"[INFO] 共有 {len(user_reviews)} 个用户有评论")

# ======================================================
# 4. 提取高质量关键词（TF-IDF）
# ======================================================
print("[INFO] 提取用户评论关键词 (TF-IDF)...")

# 连接每个用户的所有评论
user_texts = [' '.join(v) for v in user_reviews.values()]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=5)
tfidf_matrix = vectorizer.fit_transform(user_texts)
vocab = np.array(vectorizer.get_feature_names_out())

user_ids = list(user_reviews.keys())
user_ground_truth = {}

for idx, uid in enumerate(user_ids):
    user_ground_truth[uid] = set(vocab[tfidf_matrix[idx].toarray().flatten() > 0])

print(f"[INFO] 示例 Gu (前3个用户关键词):")
for i, uid in enumerate(list(user_ground_truth.keys())[:3]):
    print(uid, list(user_ground_truth[uid])[:10])

# ======================================================
# 5. 从 PGPR 路径中提取模型解释的词 (Su)
# ======================================================
def extract_words_from_path(path):
    words = []
    for rel, ent_type, ent_id in path:
        if 'word' in ent_type or rel in ['described_as', 'mentions']:
            words.append(f"{ent_type}_{ent_id}")
    return words

user_explanations = defaultdict(set)
for path in paths:
    user_id = path[0][2] if path[0][1] == 'user' else None
    if user_id is not None:
        user_explanations[user_id].update(extract_words_from_path(path))

print(f"[INFO] 共有 {len(user_explanations)} 个用户生成了解释路径")

# ======================================================
# 6. 计算 Precision / Recall / F1
# ======================================================
precision_list, recall_list, f1_list = [], [], []

for uid in user_explanations.keys():
    Su = user_explanations[uid]
    Gu = user_ground_truth.get(uid, set())
    if len(Su) == 0 or len(Gu) == 0:
        continue

    intersection = len(Su.intersection(Gu))
    P = intersection / (len(Su) + 1)
    R = intersection / (len(Gu) + 1)
    F1 = 2 * P * R / (P + R + 1e-9)

    precision_list.append(P)
    recall_list.append(R)
    f1_list.append(F1)

print("\n========== 可解释性指标 ==========")
print("平均 Precision:", np.mean(precision_list))
print("平均 Recall:", np.mean(recall_list))
print("平均 F1:", np.mean(f1_list))

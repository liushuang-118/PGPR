import pickle
import gzip
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# ===== 1️⃣ 加载 PGPR 生成的 reasoning paths =====
path_file = './tmp/Amazon_Beauty/train_agent/policy_paths_epoch1.pkl'
with open(path_file, 'rb') as f:
    data = pickle.load(f)
print(f"[INFO] 已加载 {len(data['paths'])} 条 reasoning paths")

# ===== 2️⃣ 加载用户映射文件 =====
user2id, id2user = {}, {}
with gzip.open('/content/PGPR/data/Amazon_Beauty/users.txt.gz', 'rt') as f:
    for i, line in enumerate(f):
        uid = line.strip()
        user2id[uid] = i
        id2user[i] = uid

# ===== 3️⃣ 提取路径中每个用户的解释词 =====
user_explanations = defaultdict(set)
for path in data['paths']:
    if path[0][1] == 'user':
        user_int_id = path[0][2]
        user_id = id2user.get(user_int_id)
        if user_id:
            for rel, ent_type, ent_id in path:
                if ent_type == 'word':
                    user_explanations[user_id].add(str(ent_id))

print(f"[INFO] 共有 {len(user_explanations)} 个用户生成了解释路径")

# ===== 4️⃣ 加载 Amazon Reviews (Beauty) =====
print("[INFO] 加载 Amazon Reviews-2023 (Beauty)...")
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
reviews = dataset['full']

# ===== 5️⃣ 计算每个用户的 TF-IDF 高权重词 =====
user_reviews = defaultdict(list)
for r in tqdm(reviews, desc="Processing reviews"):
    user_reviews[r['user_id']].append(r['text'])

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
texts = [' '.join(v) for v in user_reviews.values()]
tfidf_matrix = vectorizer.fit_transform(texts)
feature_names = np.array(vectorizer.get_feature_names_out())

user_ground_truth = {}
for i, (uid, docs) in enumerate(user_reviews.items()):
    top_idx = np.argsort(tfidf_matrix[i].toarray()[0])[-10:]  # top 10 keywords
    user_ground_truth[uid] = set(feature_names[top_idx])

print(f"[INFO] 共有 {len(user_ground_truth)} 个用户有评论")

# ===== 6️⃣ 计算 Precision / Recall / F1 =====
precisions, recalls, f1s = [], [], []

for uid, G in user_ground_truth.items():
    E = user_explanations.get(uid, set())
    if not E or not G:
        continue
    inter = len(E & G)
    p = inter / len(E)
    r = inter / len(G)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)

print("\n========== 可解释性指标 ==========")
print(f"平均 Precision: {np.nanmean(precisions):.4f}")
print(f"平均 Recall:    {np.nanmean(recalls):.4f}")
print(f"平均 F1:        {np.nanmean(f1s):.4f}")

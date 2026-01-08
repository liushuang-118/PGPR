import os
import gzip
import pickle
import csv

# ---------------- 配置 ----------------
pgpr_data_dir = r"D:\PGPR-RippleNet\data\Amazon_Beauty"
kg_pkl_path = r"D:\PGPR-RippleNet\tmp\Amazon_Beauty\kg.pkl"
output_dir = os.path.join(pgpr_data_dir, "ripplenet_data")
os.makedirs(output_dir, exist_ok=True)

kg_rehashed_path = os.path.join(output_dir, "kg_rehashed.txt")
item_mapping_path = os.path.join(output_dir, "item_index2entity_id_rehashed.txt")
entity2global_path = os.path.join(output_dir, "entity2global_id.txt")

# ---------------- 1. 读取实体文件 ----------------
def read_gz_list(file_name):
    path = os.path.join(pgpr_data_dir, file_name)
    res = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            val = line.strip()
            if val:
                res.append(val)
    return res

users = read_gz_list("users.txt.gz")
products = read_gz_list("product.txt.gz")
related_products = read_gz_list("related_product.txt.gz")
brands = read_gz_list("brand.txt.gz")
categories = read_gz_list("category.txt.gz")
words = read_gz_list("vocab.txt.gz")

# ---------------- 2. 构建全局 ID ----------------
entity2id = {}
current_id = 0

# ---------- product（ASIN）----------
item2id = {}
all_products = products + related_products
for asin in all_products:
    if asin not in item2id:
        item2id[asin] = current_id
        current_id += 1

# 保存 item_index2entity_id_rehashed.txt（用制表符）
with open(item_mapping_path, "w", encoding="utf-8") as f:
    for asin, gid in item2id.items():
        f.write(f"{asin}\t{gid}\n") 

# ---------- 其他实体 ----------
def assign_ids(lst, prefix):
    global current_id
    id_map = {}
    for idx, _ in enumerate(lst):
        key = f"{prefix}:{idx}"
        entity2id[key] = current_id
        id_map[idx] = current_id
        current_id += 1
    return id_map

user2id = assign_ids(users, "user")
brand2id = assign_ids(brands, "brand")
category2id = assign_ids(categories, "category")
word2id = assign_ids(words, "word")

print(f"✓ 全局 ID 分配完成，总数: {current_id}")

# ---------------- 3. 保存【可读版】entity2global_id.txt ----------------
with open(entity2global_path, "w", encoding="utf-8") as f:
    f.write("entity_type\tentity_value\toriginal_id\tglobal_id\n")

    # product
    for idx, asin in enumerate(products):
        gid = item2id.get(asin)
        if gid is not None:
            f.write(f"product\t{asin}\t{idx}\t{gid}\n")

    offset = len(products)
    for idx, asin in enumerate(related_products):
        gid = item2id.get(asin)
        if gid is not None:
            f.write(f"product\t{asin}\t{idx + offset}\t{gid}\n")

    # user
    for idx, val in enumerate(users):
        gid = user2id.get(idx)
        f.write(f"user\t{val}\t{idx}\t{gid}\n")

    # brand
    for idx, val in enumerate(brands):
        gid = brand2id.get(idx)
        f.write(f"brand\t{val}\t{idx}\t{gid}\n")

    # category
    for idx, val in enumerate(categories):
        gid = category2id.get(idx)
        f.write(f"category\t{val}\t{idx}\t{gid}\n")

    # word
    for idx, val in enumerate(words):
        gid = word2id.get(idx)
        f.write(f"word\t{val}\t{idx}\t{gid}\n")

print(f"✓ entity2global_id.txt 已保存（可读版）")

# ---------------- 4. 加载 PGPR KG ----------------
with open(kg_pkl_path, "rb") as f:
    kg = pickle.load(f)

# ---------------- 5. 商品 index → ASIN ----------------
product_index2asin = {}
for i, asin in enumerate(products):
    product_index2asin[i] = asin

offset = len(products)
for i, asin in enumerate(related_products):
    product_index2asin[i + offset] = asin

# ---------------- 6. 获取全局 ID ----------------
def get_global_id(entity_type, orig_id):
    if entity_type == "product":
        asin = product_index2asin.get(orig_id)
        return item2id.get(asin)
    else:
        return entity2id.get(f"{entity_type}:{orig_id}")

# ---------------- 7. 关系 → 尾实体类型 ----------------
def determine_tail_type(head_type, relation):
    r = relation.lower()
    if head_type == "user":
        if "purchase" in r:
            return "product"
        if "mention" in r:
            return "word"
    elif head_type == "product":
        if "described" in r:
            return "word"
        if "produced" in r:
            return "brand"
        if "belong" in r:
            return "category"
        if "also" in r or "together" in r:
            return "product"
    return "product"

# ---------------- 8. 构建 kg_rehashed.txt ----------------
triples = []

for head_type in kg.G:
    for head_id in kg.G[head_type]:
        for relation, tail_ids in kg.G[head_type][head_id].items():
            if not isinstance(tail_ids, (list, tuple, set)):
                tail_ids = [tail_ids]
            for tail_id in tail_ids:
                tail_type = determine_tail_type(head_type, relation)
                h = get_global_id(head_type, head_id)
                t = get_global_id(tail_type, tail_id)
                if h is not None and t is not None:
                    triples.append((h, relation, t))

print(f"✓ 三元组数量: {len(triples)}")

with open(kg_rehashed_path, "w", encoding="utf-8") as f:
    for h, r, t in triples:
        f.write(f"{h}\t{r}\t{t}\n")  

print("✓ kg_rehashed.txt 生成完成")
print("✓ item_index2entity_id_rehashed.txt 生成完成")
print("✓ entity2global_id.txt 生成完成（最终版）")


# ---------------- 9. 生成 Rating.csv ----------------
rating_csv_path = os.path.join(output_dir, "Rating.csv")
raw_rating_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\All_Beauty.csv"

# reviewerID → 行号
user_value2index = {u: i for i, u in enumerate(users)}

cnt_total = 0
cnt_written = 0

with open(raw_rating_path, "r", encoding="utf-8") as fin, \
     open(rating_csv_path, "w", encoding="utf-8") as fout:

    # 固定表头（你要求的格式）
    fout.write('"User-ID";"ISBN";"Book-Rating"\n')

    for line in fin:
        cnt_total += 1
        line = line.strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) != 4:
            continue

        asin, reviewer_id, rating, _ = parts

        # reviewerID → user 行号
        if reviewer_id not in user_value2index:
            continue

        user_orig_id = user_value2index[reviewer_id]
        user_global_id = entity2id.get(f"user:{user_orig_id}")

        if user_global_id is None:
            continue

        # 写入一行
        fout.write(f'"{user_global_id}";"{asin}";"{rating}"\n')
        cnt_written += 1

print(f"Rating.csv 生成完成: {rating_csv_path}")
print(f"原始评分行数: {cnt_total}")
print(f"成功写入行数: {cnt_written}")


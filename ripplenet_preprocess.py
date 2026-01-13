import os
import gzip
import csv

# ---------------- 配置 ----------------
data_dir = r"D:\PGPR-RippleNet\data\Amazon_Beauty\ripplenet_data"
train_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\train.txt.gz"
test_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\test.txt.gz"
entity2global_path = os.path.join(data_dir, "entity2global_id.txt")
output_path = os.path.join(data_dir, "ratings_final.csv")

# ---------------- 1. 读取 entity2global_id.txt ----------------
lookup = {}
with open(entity2global_path, "r", encoding="utf-8") as f:
    next(f)  # skip header
    for line in f:
        entity_type, entity_value, original_id, global_id = line.strip().split("\t")
        lookup[(entity_type, int(original_id))] = (entity_value, int(global_id))

print(f"[Info] lookup 表大小: {len(lookup)}")

# ---------------- 2. 读取 train/test 并收集前两个值 ----------------
def read_first_two_ids(gz_path):
    pairs = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            user_id = int(parts[0])
            product_id = int(parts[1])
            pairs.append((user_id, product_id))
    return pairs

train_pairs = read_first_two_ids(train_path)
test_pairs = read_first_two_ids(test_path)
all_pairs = train_pairs + test_pairs
print(f"[Info] 总共收集的 user-product 对: {len(all_pairs)}")

# ---------------- 3. 写入 CSV，所有字段加双引号 ----------------
with open(output_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
    writer.writerow(["User-ID", "ISBN", "Book-Rating"])  # 表头

    missing_count = 0
    for user_orig, prod_orig in all_pairs:
        user_info = lookup.get(("user", user_orig), None)
        prod_info = lookup.get(("product", prod_orig), None)

        if user_info is None or prod_info is None:
            missing_count += 1
            continue

        user_global_id = str(user_info[1])   # global_id
        product_value = prod_info[0]         # entity_value
        rating = "1"                         # 所有评分为 1

        writer.writerow([user_global_id, product_value, rating])

print(f"[Info] 写入完成: {output_path}")
print(f"[Info] 未找到映射的条目数量: {missing_count}")

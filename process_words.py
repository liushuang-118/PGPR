import os
import gzip
import csv

# ---------------- 配置 ----------------
pgpr_data_dir = r"D:\PGPR-RippleNet\data\Amazon_Clothing"
output_file = os.path.join(pgpr_data_dir, "ripplenet_data", "train_test_global.txt.gz")
train_file = os.path.join(pgpr_data_dir, "train.txt.gz")
test_file = os.path.join(pgpr_data_dir, "test.txt.gz")
entity2global_path = os.path.join(pgpr_data_dir, "ripplenet_data", "entity2global_id.txt")

# ---------------- 1. 读取 entity2global_id.txt ----------------
user2id = {}
word2id = {}

with open(entity2global_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        etype = row["entity_type"]
        original_id = int(row["original_id"])
        global_id = int(row["global_id"])
        if etype == "user":
            user2id[original_id] = global_id
        elif etype == "word":
            word2id[original_id] = global_id

print(f"✓ 读取全局 ID 映射完成，用户 {len(user2id)}, 单词 {len(word2id)}")

# ---------------- 2. 处理 train/test 文件 ----------------
def convert_file_to_global(file_path, user2id, word2id):
    """
    将 train/test 文件 user_id 和 word_id 替换为全局 ID
    每行: user item word1 word2 ...
    返回: list of str (lines)
    """
    lines_out = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            old_user = int(parts[0])
            item = parts[1]  # item 保持不变
            word_ids = [int(w) for w in parts[2:]]

            # 替换 user_id 和 word_id
            new_user = user2id.get(old_user, old_user)
            new_words = [str(word2id.get(w, w)) for w in word_ids]

            line_new = " ".join([str(new_user), item] + new_words)
            lines_out.append(line_new)
    return lines_out

# ---------------- 3. 转换并合并 ----------------
train_lines = convert_file_to_global(train_file, user2id, word2id)
test_lines = convert_file_to_global(test_file, user2id, word2id)
all_lines = train_lines + test_lines

# ---------------- 4. 写入新文件 ----------------
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with gzip.open(output_file, "wt", encoding="utf-8") as f:
    for line in all_lines:
        f.write(line + "\n")

print(f"✓ 合并完成，已保存为 {output_file}")


# import gzip

# file_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\ripplenet_data\train_test_global.txt.gz"

# with gzip.open(file_path, "rt", encoding="utf-8") as f:
#     for i, line in enumerate(f):
#         print(line.strip())
#         if i >= 9:  # 显示前 10 行
#             break


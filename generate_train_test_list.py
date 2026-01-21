import os
import gzip
from collections import defaultdict

# ---------------- 配置 ----------------
data_dir = r"D:\PGPR-KGAT\data\Amazon_Beauty"
kgat_dir = os.path.join(data_dir, "kgat_data")
os.makedirs(kgat_dir, exist_ok=True)

train_gz = os.path.join(data_dir, "train.txt.gz")
test_gz = os.path.join(data_dir, "test.txt.gz")

train_out = os.path.join(kgat_dir, "train.txt")
test_out = os.path.join(kgat_dir, "test.txt")

# ---------------- 核心函数 ----------------
def convert_and_aggregate(input_gz, output_txt):
    user_items = defaultdict(list)

    with gzip.open(input_gz, "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_id = parts[0]
            item_id = parts[1]
            user_items[user_id].append(item_id)

    with open(output_txt, "w", encoding="utf-8") as f:
        for user_id, items in user_items.items():
            # item 之间用空格分隔
            f.write(f"{user_id} {' '.join(items)}\n")

    print(f"✓ 生成完成: {output_txt}（用户数 {len(user_items)}）")

# ---------------- 执行 ----------------
convert_and_aggregate(train_gz, train_out)
convert_and_aggregate(test_gz, test_out)

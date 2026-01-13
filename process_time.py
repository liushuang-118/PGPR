import os
import pandas as pd

# ---------------- 配置 ----------------
train_path = r"D:\Thesis_Project\Models\RippleNet\data\book\time_train.csv"
test_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\time_test.csv"
entity2global_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\ripplenet_data\entity2global_id.txt"
output_path = r"D:\PGPR-RippleNet\data\Amazon_Beauty\ripplenet_data\time_all_global.csv"

# ---------------- 1. 读取 UID → global_id 映射 ----------------
print("读取 entity2global_id.txt 构建 UID 映射...")
entity2global = pd.read_csv(entity2global_path, sep="\t")
user_map = entity2global[entity2global['entity_type'] == 'user'][['original_id', 'global_id']]
user_map_dict = dict(zip(user_map['original_id'], user_map['global_id']))

# ---------------- 2. 读取 train/test 文件 ----------------
print("读取 CSV 文件...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 检查列名是否正确
print("train_df 列:", train_df.columns)
print("test_df 列:", test_df.columns)

# ---------------- 3. 替换 UID 为全局 ID ----------------
print("转换 UID 为全局 ID ...")
train_df['UID'] = train_df['UID'].map(user_map_dict)
test_df['UID'] = test_df['UID'].map(user_map_dict)

# 如果有 UID 未匹配上，提示
missing_train = train_df['UID'].isna().sum()
missing_test = test_df['UID'].isna().sum()
if missing_train > 0 or missing_test > 0:
    print(f"⚠ 有未匹配的 UID: train={missing_train}, test={missing_test}")

# ---------------- 4. 合并文件 ----------------
all_df = pd.concat([train_df, test_df], ignore_index=True)

# ---------------- 5. 保存 ----------------
all_df.to_csv(output_path, index=False)
print(f"✓ 合并并替换 UID 完成，保存到 {output_path}")

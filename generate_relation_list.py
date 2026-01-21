import os

# 输出目录
kgat_dir = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data"
os.makedirs(kgat_dir, exist_ok=True)

relation_list_path = os.path.join(kgat_dir, "relation_list.txt")

# 关系定义（顺序即 remap_id）
relations = [
    "described_as",
    "produced_by",
    "belongs_to",
    "also_bought",
    "also_viewed",
    "bought_together"
]

with open(relation_list_path, "w", encoding="utf-8") as f:
    f.write("org_id\tremap_id\n")
    for remap_id, rel in enumerate(relations):
        f.write(f"{rel}\t{remap_id}\n")

print(f"生成完成: {relation_list_path}")

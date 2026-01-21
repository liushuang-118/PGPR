# -*- coding: utf-8 -*-
input_file = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data\entity2global_id.txt"
item_output_file = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data\item_list.txt"
entity_output_file = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data\entity_list.txt"

item_list = []     # 存放 product 类型的 (entity_value, global_id)
entity_list = []   # 存放非 user 类型的 global_id

with open(input_file, "r", encoding="utf-8") as f:
    next(f)  # 跳过表头
    for line in f:
        line = line.strip()
        if not line:
            continue
        entity_type, entity_value, original_id, global_id = line.split("\t")
        
        # 生成 item_list.txt (仅 product)
        if entity_type == "product":
            item_list.append((entity_value, global_id))
        
        # 生成 entity_list.txt (除 user 外)
        if entity_type != "user":
            entity_list.append(global_id)

# 写入 item_list.txt
with open(item_output_file, "w", encoding="utf-8") as f:
    f.write("org_id\tremap_id\tfreebase_id\n")  # 表头
    for remap_id, (org_id, freebase_id) in enumerate(item_list):
        f.write(f"{org_id}\t{remap_id}\t{freebase_id}\n")

# 写入 entity_list.txt
with open(entity_output_file, "w", encoding="utf-8") as f:
    f.write("org_id\tremap_id\n")  # 表头
    for remap_id, org_id in enumerate(entity_list):
        f.write(f"{org_id}\t{remap_id}\n")

print(f"生成成功：\n- item_list.txt -> {item_output_file}\n- entity_list.txt -> {entity_output_file}")

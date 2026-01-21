import pickle

# -----------------------------
# 路径
# -----------------------------
kg_file = r"D:\PGPR-RippleNet\tmp\Amazon_Beauty\kg.pkl"
entity_map_file = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data\entity2global_id.txt"
relation_list_file = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data\relation_list.txt"
output_file = r"D:\PGPR-KGAT\data\Amazon_Beauty\kgat_data\kg_final.txt"

# -----------------------------
# 1. 构建 entity_map
# -----------------------------
entity_map = {}
with open(entity_map_file, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        entity_type, entity_value, original_id, global_id = line.strip().split("\t")
        entity_map[(entity_type, int(original_id))] = int(global_id)

# -----------------------------
# 2. 构建 relation_map (name -> remap_id)
# -----------------------------
relation_map = {}
with open(relation_list_file, "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        org_id, remap_id = line.strip().split("\t")
        relation_map[org_id] = int(remap_id)

# -----------------------------
# 3. 加载 KnowledgeGraph
# -----------------------------
with open(kg_file, "rb") as f:
    kg = pickle.load(f)

# -----------------------------
# 4. 设置需要过滤的关系名
# -----------------------------
filtered_relations = {'purchase', 'mentions'}

# -----------------------------
# 5. 设置关系对应尾实体类型
# -----------------------------
relation_tail_type = {
    'described_as': 'word',
    'also_bought': 'product',
    'also_viewed': 'product',
    'belongs_to': 'category',
    'produced_by': 'brand',
    'bought_together': 'product',
}

# -----------------------------
# 6. 遍历三元组，转换为 global_id 并过滤
# -----------------------------
triplets = []
skipped_count = 0

for entity_type, entities in kg.G.items():
    for local_head_id, neighbors in entities.items():
        key_head = (entity_type, int(local_head_id))
        if key_head not in entity_map:
            skipped_count += 1
            continue
        head_global_id = entity_map[key_head]

        for relation_name, tail_list in neighbors.items():
            if relation_name in filtered_relations:
                continue

            # 确定 tail 类型
            tail_type_default = relation_tail_type.get(relation_name, entity_type)

            # 获取 relation_id，如果不存在 relation_map 则跳过
            if relation_name not in relation_map:
                skipped_count += len(tail_list)
                continue
            relation_id = relation_map[relation_name]

            for local_tail_id in tail_list:
                if isinstance(local_tail_id, tuple):
                    tail_type, tail_local_id = local_tail_id
                else:
                    tail_type, tail_local_id = tail_type_default, local_tail_id

                key_tail = (tail_type, int(tail_local_id))
                if key_tail not in entity_map:
                    skipped_count += 1
                    continue
                tail_global_id = entity_map[key_tail]

                triplets.append((head_global_id, relation_id, tail_global_id))

# -----------------------------
# 7. 保存到 txt
# -----------------------------
with open(output_file, "w", encoding="utf-8") as f:
    for h, r, t in triplets:
        f.write(f"{h}\t{r}\t{t}\n")

print(f"生成完成，三元组总数: {len(triplets)}")
print(f"跳过的缺失实体或关系三元组数量: {skipped_count}")
print(f"保存路径: {output_file}")

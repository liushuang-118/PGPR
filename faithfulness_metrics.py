import pickle
from collections import defaultdict, Counter
import pandas as pd

# ======== 参数 ========
file_path = r'tmp\Amazon_Beauty\train_agent\training_paths_epoch_1.pkl'

# ======== 读取文件 ========
with open(file_path, 'rb') as f:
    data = pickle.load(f)

print(f"共有用户数量: {len(data)}")

# ======== 提取规则分布 ========
F_u = {}  # {user_id: {rule_pattern: prob}}
global_rule_set = set()

for user_id, paths in data.items():
    rule_counter = Counter()
    for path in paths:
        # 提取关系序列，忽略 self_loop
        relations = [rel for rel, entity_type, entity_id in path if rel != 'self_loop']
        if len(relations) == 0:
            continue
        rule_pattern = tuple(relations)
        rule_counter[rule_pattern] += 1
        global_rule_set.add(rule_pattern)
    
    total = sum(rule_counter.values())
    if total > 0:
        F_u[user_id] = {rule: count / total for rule, count in rule_counter.items()}
    else:
        F_u[user_id] = {}

print(f"共有不同规则模板数量: {len(global_rule_set)}")

# ======== 打印几个样例 ========
print("\n==== F(u) 样例输出（前 5 个用户） ====")
for i, (user_id, dist) in enumerate(F_u.items()):
    print(f"\n用户 {user_id}:")
    for rule, prob in dist.items():
        print(f"  规则 {rule}: 概率 = {prob:.3f}")
    if i >= 4:
        break

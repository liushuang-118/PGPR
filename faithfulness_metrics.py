import pickle
from collections import Counter

# ==============================
# 配置文件路径
# ==============================
input_pkl = r'tmp/Amazon_Beauty/train_agent/training_paths_epoch_1.pkl'
output_pkl = r'tmp/Amazon_Beauty/train_agent/F_u.pkl'
num_samples_per_user = 10  # 每个用户采样路径数，按论文描述

# ==============================
# 读取训练路径
# ==============================
with open(input_pkl, 'rb') as f:
    training_paths = pickle.load(f)

print(f"Loaded training paths for {len(training_paths)} users.")

# ==============================
# 函数：将路径转换为规则（只保留关系序列）
# ==============================
def path_to_rule(path):
    # path: [(relation, node_type, node_id), ...]
    return tuple([step[0] for step in path])

# ==============================
# 计算 F(u)
# ==============================
F_u_dict = dict()

for u, paths in training_paths.items():
    # 如果路径太多，随机采样 num_samples_per_user 条
    if len(paths) > num_samples_per_user:
        import random
        paths = random.sample(paths, num_samples_per_user)

    rules = [path_to_rule(p) for p in paths]
    count = Counter(rules)
    total = sum(count.values())
    F_u = {rule: freq / total for rule, freq in count.items()}  # 归一化成概率
    F_u_dict[u] = F_u

print(f"Computed F(u) for {len(F_u_dict)} users.")


# ==============================
# 打印一个例子
# ==============================
example_user = list(F_u_dict.keys())[0]
print(f"Example F(u) for user {example_user}:")
for i, (rule, prob) in enumerate(F_u_dict[example_user].items()):
    print(f"  Rule {i+1}: {rule} -> {prob:.4f}")
    if i >= 4:  # 只展示前 5 条
        break

# # ==============================
# # 保存结果
# # ==============================
# with open(output_pkl, 'wb') as f:
#     pickle.dump(F_u_dict, f)

# print(f"Saved F(u) to {output_pkl}")

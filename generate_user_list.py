import os
import gzip

# 数据路径
data_dir = r"D:\PGPR-KGAT\data\Amazon_Beauty"
input_file = os.path.join(data_dir, "users.txt.gz")

# 输出路径
kgat_dir = os.path.join(data_dir, "kgat_data")
os.makedirs(kgat_dir, exist_ok=True)
output_file = os.path.join(kgat_dir, "user_list.txt")

# 读取用户并生成 remap_id
with gzip.open(input_file, 'rt', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    # 写表头
    f_out.write("org_id\tremap_id\n")
    
    for idx, line in enumerate(f_in):
        org_id = line.strip()
        if org_id:  # 忽略空行
            f_out.write(f"{org_id} {idx}\n")

print(f"生成完成: {output_file}")

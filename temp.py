import hashlib
import base64

def get_counter_key(i):
    hash_bytes = hashlib.sha256(str(i).encode('utf-8')).digest()
    counter_str = base64.urlsafe_b64encode(hash_bytes).decode('utf-8')[:8]

    return counter_str

for i in range(20):
    counter_key = get_counter_key(i)
    print(f"python main.py --text \"jasb eroigsdo hahsdsjkbvew {counter_key}{counter_key}\"")

# def remove_long_lines(file_path, output_path, max_columns=2500):
#     with open(file_path, 'r', encoding='utf-8') as infile:
#         lines = infile.readlines()
    
#     # 过滤掉字符数超过 max_columns 的行
#     filtered_lines = [line for line in lines if len(line.strip()) <= max_columns]
    
#     # 将过滤后的内容写入新文件
#     with open(output_path, 'w', encoding='utf-8') as outfile:
#         outfile.writelines(filtered_lines)

# # 调用函数
# input_file = 'financial_data.txt'  # 输入文件路径
# output_file = 'financial_data_modified.txt'  # 输出文件路径
# remove_long_lines(input_file, output_file)



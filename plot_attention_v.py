import sys
import os

# 获取本地库的路径
local_transformers_path = "transformers-4.37.2"

# # 将本地库的路径添加到 sys.path
# sys.path.insert(0, local_transformers_path)

sys.path.append(local_transformers_path)

# 验证是否使用了本地库
import transformers-4.37.2 as transformers
print(transformers.__file__)  # 这应该打印出本地库的路径
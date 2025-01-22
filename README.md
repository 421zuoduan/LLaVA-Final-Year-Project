# My Document for Final Year Project

首先依照 `README_LLaVA.md` 配环境, flash-attn 安装要编译很长时间.

LLaVA-1.5 模型参数已经部署在 `/home/cuiruochen/model/llava-v1.5-7b` 路径下, Vision Encoder 模型参数部署在 `/home/cuiruochen/model/clip-vit-large-patch14-336`

明确目标: **可视化LLaVA-1.5, 做多模态大模型的不确定估计**

该仓库用于**尝试**提出一种不确定估计的改进方法. 此外还 Fork  `Lm-Polygraph` 来进行不同方法的比较. 目标是在 LLaVA 上跑通测试流程, 将自己的方法在两个仓库上都实现

## 可视化 Attention Values

## Semantic Entropy 跑通

semantic entropy 仓库的 `generate_ans.py` 生成答案并保存在 pkl 文件中, 然后调用 `compute_uncertainty_measures.py` 读取 pkl 文件并计算语义熵. llava 跑 pope 调用了 `llava/eval/model_vqa_loader.py`

考虑到 semantic entropy 原本的代码扩展性比较差, 数据集也读取的不多, 所以我最好在 llava 原本的代码上改, 也就是 model_vqa_loader.py; 至于X改过的语义熵, 复用性不高, 还是不借鉴了吧.





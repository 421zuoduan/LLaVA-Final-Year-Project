# My Document for Final Year Project

首先依照 `README_LLaVA.md` 配环境, flash-attn 安装要编译很长时间.

LLaVA-1.5 模型参数已经部署在 `/home/cuiruochen/model/llava-v1.5-7b` 路径下, Vision Encoder 模型参数部署在 `/home/cuiruochen/model/clip-vit-large-patch14-336`

明确目标: **可视化LLaVA-1.5, 做多模态大模型的不确定估计**

该仓库用于**尝试**提出一种不确定估计的改进方法. 此外还 Fork  `Lm-Polygraph` 来进行不同方法的比较. 目标是在 LLaVA 上跑通测试流程, 将自己的方法在两个仓库上都实现

## 可视化 Attention Values

## Semantic Entropy 跑通

semantic entropy 仓库的 `generate_ans.py` 生成答案并保存在 pkl 文件中, 然后调用 `compute_uncertainty_measures.py` 读取 pkl 文件并计算语义熵. llava 跑 pope 调用了 `llava/eval/model_vqa_loader.py`

考虑到 semantic entropy 原本的代码扩展性比较差, 数据集也读取的不多, 所以我最好在 llava 原本的代码上改, 也就是 model_vqa_loader.py; 至于X改过的语义熵, 复用性不高, 还是不借鉴了吧.

### 更改思路

将各个数据集原本的代码里采样过程用 for 循环的方法改成多次多项式采样, 并将结果保存在 pkl 文件中, 保存的代码用语义熵的 `generate_answers.py`; 然后调用语义熵的 `compute_uncertainty_measures.py` 读取 pkl 文件并计算语义熵.


### model_vqa_loader.py

pope 的脚本命令是

```
python -m semantic_entropy.model_vqa_loader \
    --model-path /home/wuzongqian/model/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl
```

这里 `python -m semantic_entropy.model_vqa_loader` 是一种运行模块的方式, 在 `sys.path` 查找 `semantic_entropy` 包中的文件; 如果 `model_vqa_loader` 中有主函数, 即执行主函数


### 测试样本抽取

pope 的问题数据来自 `playground/data/eval/pope/llava_pope_test.jsonl`, 其中 1-3000 行为 adversarial 数据, 3001-5910 行为 random 数据, 5911-8910 为 popular 数据. 全部近 9000 条样本跑一遍要 45min, 采样 3 次要 2h. 计划每一类取 10 个样本先把代码跑通, 然后每一类取 200 或 500 个样本
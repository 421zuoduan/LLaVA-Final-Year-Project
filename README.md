# My Document for Final Year Project

首先依照 `README_LLaVA.md` 配环境, flash-attn 安装要编译很长时间.

LLaVA-1.5 模型参数已经部署在 `/home/cuiruochen/model/llava-v1.5-7b` 路径下, Vision Encoder 模型参数部署在 `/home/cuiruochen/model/clip-vit-large-patch14-336`

明确目标: **可视化LLaVA-1.5, 做多模态大模型的不确定估计**

该仓库用于**尝试**提出一种不确定估计的改进方法. 此外还 Fork  `Lm-Polygraph` 来进行不同方法的比较. 目标是在 LLaVA 上跑通测试流程, 将自己的方法在两个仓库上都实现

## 可视化 Attention Values

## Semantic Entropy + LLaVA 跑通

semantic entropy 仓库的 `generate_ans.py` 生成答案并保存在 pkl 文件中, 然后调用 `compute_uncertainty_measures.py` 读取 pkl 文件并计算语义熵. llava 跑 pope 调用了 `llava/eval/model_vqa_loader.py`

考虑到 semantic entropy 原本的代码扩展性比较差, 数据集也读取的不多, 所以我最好在 llava 原本的代码上改, 也就是 model_vqa_loader.py; ~~至于X改过的语义熵, 复用性不高, 还是不借鉴了吧.~~

### 更改思路

将各个数据集原本的代码里采样过程用 for 循环的方法改成多次多项式采样, ~~并将结果保存在 pkl 文件中, 保存的代码用语义熵的 `generate_answers.py`; 然后调用语义熵的 `compute_uncertainty_measures.py` 读取 pkl 文件并计算语义熵.~~, model.generate 的 outputs.sequence 可以进一步得到 logit, 从而在 `model_vqa_loader.py` 计算语义熵, 无需更改 transformers 库, 也无需本地保存 pkl 文件.

计算出语义熵后, 想得到 AUROC, 需要先进行贪婪解码得到答案, 然后根据问题ID从 `playground/data/eval/pope` 路径下文件中提取标签, 从而得到 `validation_is_false`, 解答错误为1, 否则为0. 然后通过 AUROC 公式计算值


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

### 计算语义熵

经检查, `compute_uncertainty_measures.py` 的代码过于复杂了, 涉及到 wandb 的使用, 需要与原有 `generate_answer.py` 代码对应, 所以不套用他的代码了, 决定自己重新写 `compute_se_myself.py`. 新文件应当实现

1. 读取 `model_vqa_loader.py` 生成的 pkl 文件, pkl 文件保存 logit, 也即 head 后经过归一化的概率
2. 读取 `model_vqa_loader.py` 生成的 json 文件, 文件格式如下
    ```
    "1": {
        "question_id": 1,
        "prompt": "Is there a snowboard in the image?\nAnswer the question using a single word or phrase.",
        "responses": [
        "No",
        "No",
        "No"
        ],
        "metadata": {
        "model": "llava-v1.5-7b"
        }
    },
    ```
3. 将 response 中的答案喂给 LLM, 让 LLM 将答案分成指定数目的类别, 将相同类别的答案的 prob 相加, 归一化后计算语义熵, 这部分原有代码实现的挺好的, 不过 xbd 的版本实现起来应该更简单, 决定读一下 xbd 的代码. xbd 在 transformers 源码里增加了语义熵计算过程, 改了 generate 函数, 我感觉这部分不用吧, 最多加个 logit 返回值的传回参数应该就够了 (yysy, 这个参数感觉也是有的). 不过纯读代码还是困难的, 那不如趁此机会重新读下 generate 源码, 看到b站有up解读, 那就拿份源码做份笔记吧.


### 源码阅读时刻:(

`model.generate` 函数在 transformers 库内的路径为 `transformers/generation/utils` 内的 GenerationMixin 类的 generate 函数, xbd 的代码也是在这里改的, xbd 用的 transformers 库是4.31.0版本, 我用的是4.37.2版本. 具体细节已经上传博客


### 计算 AUROC

语义熵给的代码, 这里 `y_score` 是置信度分数, 也即语义熵分数, y_true 是真实标签

```
def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)
```

作者回复 issue 了, `'AUROC'： [validation_is_false， measure_values]`, 所以上面的 `y_true` 是答案是否预测错误, 错误为 1; `y_score` 是语义熵的值. 


获取 validation_is_false 的逻辑:

首先明确, 每个样本贪婪解码的结果与真实标签对比, 得到 validation_is_false

1. 获取真实标签: 由于只选了一部分样本, 所以要先读取标签文件为3个列表, 在样本采样的循环内获取样本问题ID, 根据问题ID确定其分类, 做问题ID的减法后在对应列表内取真实标签
2. 
3. 获取贪婪解码结果: 贪婪解码得到的结果保存在指定文件内, 读取该文件, 在样本采样的循环内获取样本问题ID, 根据问题ID得到贪婪解码结果, 根据得到的贪婪结果解码结果与真实标签, 在样本采样的循环内计算 validation_is_false

## 结果分析

SE 论文里采样次数取10, 我取了15个样本每个采样10次的AUROC结果均为1.0; 降低采样次数时, 朴素熵的值要高于语义熵, 原因待分析

多次实验, 得到语义熵 AUROC 大于 朴素熵的结果, 取 300 个样本每个样本采样 10 次. 结果已保存在 `playground/data/eval/pope/answers/300_10samples`

## 改进语义熵计算方法

现有方法没有加入长度归一化, 计划加入 token 的长度归一化

语义熵加偏移值?

## 调用豆包大模型

从火山引擎调用豆包的api, 先装火山引擎的库. 不过也可以使用 OpenAI 的代码, 这边兼容了

```
pip install 'volcengine-python-sdk[ark]'
```

实现内容为:

1. 在所有数据进行第一次处理后, 将熵排序
2. 按熵排序顺序, 得到对应高熵样本的 question 和 image
3. 将 question 和 image 喂给豆包生成 response
4. response 与 label 检查是否正确, 得到 check_is_false 和新的 entropy_list
5. 将 check_is_false 和 entropy_list 替换既有 validation_is_false 和 entropy list
6. 计算新的 AUROC 和 AURAC
7. 比对新旧 AUROC, AURAC 效果

问题难点在于获取 image, 在 POPE 的代码中, 已经将经过视觉编码器的 image_tensor 与 quesition

```
for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
```


## 一些有趣的小知识

`model.generate` 有关的 `sequence_length` 不包括标志生成开始的 <bos>, 但包括标志生成结束的 <eos>, 因为后者要计算概率, 对应到词表里

`model.generate` 函数的 `output_scores` 参数设置返回的 `output_ids.scores` 是未归一化的 logit, 即输出层 softmax 前的那部分 embedding, 但是这里没有 <eos> 的 logit
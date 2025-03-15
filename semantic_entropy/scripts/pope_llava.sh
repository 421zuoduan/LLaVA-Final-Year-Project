# 测试 150 样本时语义熵和朴素熵的情况
python -m semantic_entropy.model_vqa_loader_pope \
    --model-path /home/wuzongqian/model/llava-v1.5-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test_300.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
    --greedy-search-results-file ./playground/data/eval/pope/answers/greedy_search/llava-v1.5-7b.jsonl \
    --annotation-dir ./playground/data/eval/pope/coco \
    --pkl-folder ./playground/data/eval/pope/answers \
    --samples 10\
    --temperature 1 \
    --conv-mode vicuna_v1




# # 测试15样本能不能跑
# python -m semantic_entropy.model_vqa_loader \
#     --model-path /home/wuzongqian/model/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test_60.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
#     --greedy-search-results-file ./playground/data/eval/pope/answers/greedy_search/llava-v1.5-7b.jsonl \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --pkl-folder ./playground/data/eval/pope/answers \
#     --samples 10\
#     --temperature 1 \
#     --conv-mode vicuna_v1

# # 测试6样本能不能跑
# python -m semantic_entropy.model_vqa_loader \
#     --model-path /home/wuzongqian/model/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test_6.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
#     --greedy-search-results-file ./playground/data/eval/pope/answers/greedy_search/llava-v1.5-7b.jsonl \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --pkl-folder ./playground/data/eval/pope/answers \
#     --samples 10\
#     --temperature 1 \
#     --conv-mode vicuna_v1


# 多次采样实现, 保存pkl实现, 保存格式待检验, 本地json保存需修改, 下面要抽样本缩短时间
# python -m semantic_entropy.model_vqa_loader \
#     --model-path /home/wuzongqian/model/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
#     --temperature 1 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl

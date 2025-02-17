# # 测试15样本能不能跑
# python -m semantic_entropy.model_vqa_loader \
#     --model-path /home/wuzongqian/model/llava-v1.5-7b \
#     --question-file ./playground/data/eval/pope/llava_pope_test_15.jsonl \
#     --image-folder ./playground/data/eval/pope/val2014 \
#     --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b.jsonl \
#     --temperature 1 \
#     --conv-mode vicuna_v1

# 测试能否从 pkl 文件提取数据计算语义熵
python compute_uncertainty_measures.py \
    --model gpt3 \
    --num_generations 3 \
    --output_train train_generations.pkl \
    --output_validation validation_generations.pkl









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

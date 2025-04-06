#!/bin/bash

python -m semantic_entropy.model_vqa_loader_mme \
    --model-path /home/wuzongqian/model/llava-v1.5-7b \
    --question-file ./playground/data/eval/MME/llava_mme_test_336.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-7b.jsonl \
    --greedy-search-results-file ./playground/data/eval/MME/answers/greedy_search/llava-v1.5-7b.jsonl \
    --annotation-dir ./playground/data/eval/MME/MME_Benchmark_release_version \
    --pkl-folder ./playground/data/eval/MME/answers \
    --samples 10\
    --temperature 0.5 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-7b

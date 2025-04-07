#!/bin/bash

python -m semantic_entropy.model_vqa_loader_textvqa \
    --model-path /home/wuzongqian/model/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl \
    --greedy-search-results-file ./playground/data/eval/textvqa/answers/greedy_search/llava-v1.5-7b.jsonl \
    --annotation-dir ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --pkl-folder ./playground/data/eval/textvqa/answers \
    --samples 10\
    --temperature 1 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b.jsonl

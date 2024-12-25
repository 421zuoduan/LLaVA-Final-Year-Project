devices=4,5,6,7

# CUDA_VISIBLE_DEVICES=$devices torchrun --nproc_per_node 4 --master_port 15631 semantic_entropy/pope_eval_se.py \
#     --coco_path playground/data/eval/pope/val2014 \
#     --model-path /home/wuzongqian/model/llava-v1.5-7b \
#     --set popular

CUDA_VISIBLE_DEVICES=$devices torchrun --nproc_per_node 4 --master_port 15631 semantic_entropy/pope_eval.py \
    --coco_path playground/data/eval/pope/val2014 \
    --model-path /home/wuzongqian/model/llava-v1.5-7b \
    --set popular

# python 哥们,留四张卡,跑大模型.py --size 30000 --gpus 4 --interval 0.01


# torchrun --nproc_per_node 4 --master_port 15631 post_interaction_block/models/llava-v1_5/pope_eval.py \
#     --coco_path post_interaction_block/data/coco2014 \
#     --pope_path post_interaction_block/data/POPE \
#     --model-path /home/cuiruochen/model/llava-v1.5-7b \
#     --set popular
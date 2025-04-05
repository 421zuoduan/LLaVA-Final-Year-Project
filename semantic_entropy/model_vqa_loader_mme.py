import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math
import pickle

import transformers
from typing import Tuple, Optional
from sklearn import metrics
from collections import defaultdict
from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeepSeek
from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import UncertaintyMeasures
from semantic_entropy.uncertainty.utils.doubao import predict_doubao
from semantic_entropy.uncertainty.utils.siliconflow import predict_qwen

def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = os.path.join(data_path, category)
        if not os.path.isdir(category_dir):
            continue
        if os.path.exists(os.path.join(category_dir, 'images')):
            image_path = os.path.join(category_dir, 'images')
            qa_path = os.path.join(category_dir, 'questions_answers_YN')
        else:
            image_path = qa_path = category_dir
        assert os.path.isdir(image_path), image_path
        assert os.path.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith('.txt'):
                continue
            for line in open(os.path.join(qa_path, file)):
                question, answer = line.strip().split('\t')
                GT[(category, file, question)] = answer
    return GT


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def compare_lists(list1, list2):
    max_len = max(len(list1), len(list2))
    differences = []
    both_zero = 0
    both_one = 0
    zero_one = 0
    one_zero = 0
    origin_right = 0
    revised_right = 0
    
    for i in range(max_len):
        val1 = list1[i] if i < len(list1) else None
        val2 = list2[i] if i < len(list2) else None
        
        if val1 == 0:
            origin_right += 1
        if val2 == 0:
            revised_right += 1
        
        # 记录差异
        if val1 != val2:
            differences.append((i, val1, val2))
        
        # 仅当两个索引都有效时统计四类情况
        if i < len(list1) and i < len(list2):
            if val1 == 0 and val2 == 0:
                both_zero += 1
            elif val1 == 1 and val2 == 1:
                both_one += 1
            elif val1 == 0 and val2 == 1:
                zero_one += 1
            elif val1 == 1 and val2 == 0:
                one_zero += 1
                
    origin_acc = origin_right / max_len
    revised_acc = revised_right / max_len
    
    return differences, both_zero, both_one, zero_one, one_zero, origin_acc, revised_acc


def calculate_auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)

def calculate_aurac(entropy_list, labels):
    # 将熵值与标签组合并排序
    combined = list(zip(entropy_list, labels))
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
    
    # 计算需要排除的样本数（20%）
    total = len(combined_sorted)
    exclude_num = int(total * 0.2)
    
    # 保留剩余样本的标签
    remaining_labels = [label for (_, label) in combined_sorted[exclude_num:]]
    
    # 计算0的比例
    count_0 = sum(1 for label in remaining_labels if label == 0)
    return count_0 / len(remaining_labels) if len(remaining_labels) > 0 else 0.0

def compute_transition_scores(
    sequences: torch.Tensor,
    scores: Tuple[torch.Tensor],
    beam_indices: Optional[torch.Tensor] = None,
    normalize_logits: bool = False,
) -> torch.Tensor:
    """
    Computes the transition scores of sequences given the generation scores (and beam indices, if beam search was
    used). This is a convenient method to quicky obtain the scores of the selected tokens at generation time.

    Parameters:
        sequences (`torch.LongTensor`):
            The generated sequences. The second dimension (sequence_length) is either equal to `max_length` or
            shorter if all batches finished early due to the `eos_token_id`.
        scores (`tuple(torch.FloatTensor)`):
            Transition scores for each vocabulary token at each generation step. Beam transition scores consisting
            of log probabilities of tokens conditioned on log softmax of previously generated tokens Tuple of
            `torch.FloatTensor` with up to `max_new_tokens` elements (one element for each generated token), with
            each tensor of shape `(batch_size*num_beams, config.vocab_size)`.
        beam_indices (`torch.LongTensor`, *optional*):
            Beam indices of generated token id at each generation step. `torch.LongTensor` of shape
            `(batch_size*num_return_sequences, sequence_length)`. Only required if a `num_beams>1` at
            generate-time.
        normalize_logits (`bool`, *optional*, defaults to `False`):
            Whether to normalize the logits (which, for legacy reasons, may be unnormalized).

    Return:
        `torch.Tensor`: A `torch.Tensor` of shape `(batch_size*num_return_sequences, sequence_length)` containing
            the transition scores (logits)

    Examples:

    ```python
    >>> from transformers import GPT2Tokenizer, AutoModelForCausalLM
    >>> import numpy as np

    >>> tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
    >>> tokenizer.pad_token_id = tokenizer.eos_token_id
    >>> inputs = tokenizer(["Today is"], return_tensors="pt")

    >>> # Example 1: Print the scores for each token generated with Greedy Search
    >>> outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
    >>> transition_scores = model.compute_transition_scores(
    ...     outputs.sequences, outputs.scores, normalize_logits=True
    ... )
    >>> # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
    >>> # encoder-decoder models, like BART or T5.
    >>> input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    >>> generated_tokens = outputs.sequences[:, input_length:]
    >>> for tok, score in zip(generated_tokens[0], transition_scores[0]):
    ...     # | token | token string | logits | probability
    ...     print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    |   262 |  the     | -1.414 | 24.33%
    |  1110 |  day     | -2.609 | 7.36%
    |   618 |  when    | -2.010 | 13.40%
    |   356 |  we      | -1.859 | 15.58%
    |   460 |  can     | -2.508 | 8.14%

    >>> # Example 2: Reconstruct the sequence scores from Beam Search
    >>> outputs = model.generate(
    ...     **inputs,
    ...     max_new_tokens=5,
    ...     num_beams=4,
    ...     num_return_sequences=4,
    ...     return_dict_in_generate=True,
    ...     output_scores=True,
    ... )
    >>> transition_scores = model.compute_transition_scores(
    ...     outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
    ... )
    >>> # If you sum the generated tokens' scores and apply the length penalty, you'll get the sequence scores.
    >>> # Tip: recomputing the scores is only guaranteed to match with `normalize_logits=False`. Depending on the
    >>> # use case, you might want to recompute it with `normalize_logits=True`.
    >>> output_length = input_length + np.sum(transition_scores.numpy() < 0, axis=1)
    >>> length_penalty = model.generation_config.length_penalty
    >>> reconstructed_scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
    >>> print(np.allclose(outputs.sequences_scores, reconstructed_scores))
    True
    ```"""
    # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
    # to a beam search approach were the first (and only) beam is always selected
    if beam_indices is None:
        beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1).to(sequences.device)
        beam_indices = beam_indices.expand(-1, len(scores))

    # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
    # seq_len - input_length
    scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

    # 3. Optionally normalize the logits (across the vocab dimension)
    # 理论上这里要传进 args, 从而替换 self.config.vocab_size, 这里从简, 直接设置 vocab_size 为 llava-1.5-7b 的 32000 了
    vocab_size = 32000
    if normalize_logits:
        # scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
        scores = scores.reshape(-1, vocab_size, scores.shape[-1])
        scores = torch.nn.functional.log_softmax(scores, dim=1)
        scores = scores.reshape(-1, scores.shape[-1])

    # 4. cut beam_indices to longest beam length
    beam_indices_mask = beam_indices < 0
    max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
    beam_indices = beam_indices.clone()[:, :max_beam_length]
    beam_indices_mask = beam_indices_mask[:, :max_beam_length]

    # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
    beam_indices[beam_indices_mask] = 0

    # 6. multiply beam_indices with vocab size to gather correctly from scores
    # beam_sequence_indices = beam_indices * self.config.vocab_size
    beam_sequence_indices = beam_indices * vocab_size

    # 7. Define which indices contributed to scores
    cut_idx = sequences.shape[-1] - max_beam_length
    indices = sequences[:, cut_idx:] + beam_sequence_indices

    # 8. Compute scores
    transition_scores = scores.gather(0, indices)

    # 9. Mask out transition_scores of beams that stopped early
    transition_scores[beam_indices_mask] = 0

    return transition_scores


def check_again(data_loader, questions, entropy_list, labels, validation_is_false, original_preds):
    """处理高熵样本的二次验证流程"""
    # ====================== 1. 构建索引映射 ======================
    idx_map = {}
    question_ids = []
    idx = 0
    for (input_ids, image, image_sizes), line in zip(data_loader, questions):
        question_id = line["question_id"]
        idx_map[idx] = {
            "prompt": line["text"],
            "image": image,          # 图像数据（假设已预处理）
            "question_id": question_id      # 保留原始索引用于结果更新
        }
        idx = idx + 1
        question_ids.append(question_id)
    all_idx = idx - 1

    # ====================== 2. 排序高熵样本 ======================
    # combined = list(zip(entropy_list, labels, range(len(entropy_list))))  # 第三个元素是连续索引
    combined = list(zip(entropy_list, labels, original_preds, question_ids, range(all_idx)))
    combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
    total_samples = len(combined_sorted)
    high_entropy_num = int(total_samples * 0.20)  # 取前20%高熵样本
    print(f"total_samples: {total_samples}; high_entropy_num: {high_entropy_num}")
    
    # ====================== 3. 处理高熵样本 ======================
    updated_validation = validation_is_false.copy()  # 创建副本避免直接修改原数据
    print(f"updated_validation: {updated_validation}")
    print(f"combined_sorted: {combined_sorted}")
    
    all_is_correct = []
    
    for sample in tqdm(combined_sorted[:high_entropy_num], desc="Processing High-Entropy Samples"):
        original_entropy, true_label, original_pred, question_id, idx = sample
        data = idx_map.get(idx)
        # if not data:
        #     continue

        # try:
        # ===== 3.1 使用豆包模型进行贪婪搜索（temperature=0） =====
        doubao_response = predict_qwen(
            prompt=data["prompt"],
            image=data["image"],
            temperature=0.0  # 强制贪婪搜索
        ).strip().lower()  # 统一小写处理
        q = data["prompt"]
        print(f"idx: {idx}")
        print(f"question_id: {question_id}")
        print(f"q: {q}")
        print(f"doubao_response: {doubao_response}")
        print(f"original_pred: {original_pred}")
        print(f"label: {true_label}")
        
        # ===== 3.2 验证响应是否正确 =====
        is_correct = 0
        if (doubao_response == "yes" and true_label == "yes") or \
            (doubao_response == "no" and true_label == "no"):
            is_correct = 1
            
        all_is_correct.append(is_correct)
        
        # ===== 3.3 更新validation结果 =====
        if is_correct:
            updated_validation[idx] = 0  # 正确预测设为0
        else:
            updated_validation[idx] = 1  # 错误预测保持1

        # except Exception as e:
        #     print(f"Error processing sample {idx}: {str(e)}")
        #     # 错误时保留原validation结果（保持1）
        #     updated_validation[idx] = 1

    # ====================== 4. 返回更新后的结果 ======================
    return updated_validation, entropy_list


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, is_check=False):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.is_check = is_check

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        if not self.is_check:
            image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_path if self.is_check else image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes

def collate_fn_check(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    # image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, is_check=False):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, is_check=is_check)
    if is_check:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn_check)
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    data_loader_check = create_data_loader(questions, args.image_folder, tokenizer, None, model.config, is_check=True)
    
    # 准备保存详细结果的字典和文件路径
    json_output_path = os.path.join(os.path.dirname(answers_file), "detailed_results.json")
    generations = {}  # 存储所有问题的详细生成结果
    
    # 准备 Entailment Model
    entailment_model = EntailmentDeepSeek(entailment_cache_id=None, entailment_cache_only=False)
    
    # 创建 validation_is_false 列表
    validation_is_false = []
    
    # 获取贪婪解码的答案
    # greedy_search_answers = []
    # with open(args.greedy_search_results_file, 'r', encoding='utf-8') as f:
    #     greedy_search_answers = [json.loads(line.strip()) for line in f]
    greedy_search_answers = {}
    with open(args.greedy_search_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())  # 解析单行JSON
            # 直接将 question_id 作为键存储整个对象
            greedy_search_answers[item["question_id"]] = item 
        
    # 获取真实标签
    # for file in os.listdir(args.annotation_dir):
    #     with open(os.path.join(args.annotation_dir, file), 'r', encoding='utf-8') as f:
    #         adv_label_list = [json.loads(line.strip()) for line in f]           
    GT = get_gt(
        data_path=args.annotation_dir
    )
        
        
        
    ### 保存所有样本多次采样的结果
    # 保存样本序列, 回答与问题ID
    all_question_ids = []
    all_multi_responses = []
    all_multi_sequences = []
    labels = []
    original_pred = []
    
    # 保存样本熵的计算结果
    all_regular_entropy = []
    all_regular_entropy_rao = []
    all_semantic_entropy = []    
    all_semantic_entropy_rao = []
    all_cluster_assignment_entropy = []
    
    #保存样本熵计算的中间变量
    all_multi_log_liks = []
    all_log_liks_agg = []
    all_semantic_ids = []
    all_log_likelihood_per_semantic_id = []      
        
    

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        
        uncertainty_computer = UncertaintyMeasures(question=cur_prompt)
        
        # Initialize storage for this question
        generations[idx] = {
            "question_id": idx,
            "prompt": cur_prompt,
            "responses": [],
            "metadata": {"model": model_name}
        }
        
        ### 保存单个样本多次采样的结果
        # 保存所有生成的序列
        multi_sequences = []
        # 保存生成的自然语言文本
        multi_responses = []
        # 保存生成的log likelihood
        multi_log_liks = []

        # Generate 3 responses with sampling
        for generation_idx in range(args.samples):  # 多次多项式采样
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    output_scores=True,
                    output_hidden_states=True,
                    # output_attentions=True,
                    return_dict_in_generate=True)
                
            n_generated = len(output_ids.scores)
            
            multi_sequences.append(output_ids.sequences)
            
            output_text = tokenizer.decode(output_ids.sequences[0, 1:-1]).strip().lower()
            multi_responses.append(output_text)
            
            transition_scores = compute_transition_scores(output_ids.sequences, output_ids.scores, normalize_logits=True)
            log_likelihoods = [score.item() for score in transition_scores[0]]
            
            if len(log_likelihoods) == 1:
                # logging.warning('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                log_likelihoods = log_likelihoods[:n_generated]
                
            multi_log_liks.append(torch.tensor(log_likelihoods))
            
            # store
            generations[idx]["responses"].append(output_text)

            # 写入本地答案文件（兼容原有格式）
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": output_text,
                "answer_id": ans_id,
                "model_id": model_name,
                "generation_idx": generation_idx,  # 新增字段标识生成序号
                "metadata": {}
            }) + "\n")
            
            
        # 保存样本问题ID, 多次采样的序列结果, 多次采样的回答结果
        all_question_ids.append(idx)
        all_multi_sequences.append(multi_sequences)
        all_multi_responses.append(multi_responses)
        
        # 保存样本多次采样的 log_likelihoods
        all_multi_log_liks.append(multi_log_liks)
        
        # 输出信息
        print(f'multi_responses: {multi_responses}')
        
       
        log_liks_agg = torch.tensor([torch.mean(log_lik) for log_lik in multi_log_liks])
        #TODO: 看下 torch.mean的意思.
        all_log_liks_agg.append(log_liks_agg)
        
        ### Compute naive entropy
        regular_entropy = uncertainty_computer.predictive_entropy(log_liks_agg)
        regular_entropy_rao = uncertainty_computer.predictive_entropy_rao(log_liks_agg)
        all_regular_entropy.append(regular_entropy)
        all_regular_entropy_rao.append(regular_entropy_rao)
        print(f'regular_entropy: {regular_entropy}')
        print(f'regular_entropy_rao: {regular_entropy_rao}')
                
        
        ### Compute semantic entropy
        semantic_ids = uncertainty_computer.get_semantic_ids(
                        strings_list=multi_responses, model=entailment_model,
                        strict_entailment=True, example=uncertainty_computer.question)
        all_semantic_ids.append(semantic_ids)
        print(f'semantic_ids: {semantic_ids}')
        
        
        ### Compute dicrete entropy from cluster assignments
        # Compute entropy from frequencies of cluster assignments, namely DSE
        cluster_assignment_entropy=uncertainty_computer.cluster_assignment_entropy(semantic_ids)
        all_cluster_assignment_entropy.append(cluster_assignment_entropy)
        print(f'cluster_assignment_entropy: {cluster_assignment_entropy}')
        
        
        ### Compute semantic entropy
        log_likelihood_per_semantic_id = uncertainty_computer.logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
        all_log_likelihood_per_semantic_id.append(log_likelihood_per_semantic_id)
        print(f'log_likelihood_per_semantic_id: {log_likelihood_per_semantic_id}')
        
        # Compute semantic_entropy
        pe = uncertainty_computer.predictive_entropy(torch.tensor(log_likelihood_per_semantic_id))
        semantic_entropy = pe
        all_semantic_entropy.append(semantic_entropy)
        print(f'semantic_entropy: {semantic_entropy}')
        
        # Compute semantic_entropy_rao
        pe = uncertainty_computer.predictive_entropy_rao(torch.tensor(log_likelihood_per_semantic_id))
        semantic_entropy_rao = pe
        all_semantic_entropy_rao.append(semantic_entropy_rao)
        print(f'semantic_entropy_rao: {semantic_entropy_rao}')
        
        
        # 获取真实答案
        category = line["question_id"].split('/')[0]
        file = line['question_id'].split('/')[-1].split('.')[0] + '.txt'
        prompt = line['text']
        
        if 'Answer the question using a single word or phrase.' in prompt:
            prompt = prompt.replace('Answer the question using a single word or phrase.', '').strip()
        if 'Please answer yes or no.' not in prompt:
            prompt = prompt + ' Please answer yes or no.'
            if (category, file, prompt) not in GT:
                prompt = prompt.replace(' Please answer yes or no.', '  Please answer yes or no.')
        label = GT[category, file, prompt]
        
        # 获取贪婪搜索结果
        pred = greedy_search_answers.get(idx)
        pred = pred["text"] if pred else ""

        original_pred.append(pred)
        
        if pred == 'Yes' and label == 'Yes':
            validation_is_false.append(0)
        elif pred == 'No' and label == 'No':
            validation_is_false.append(0)
        else:
            validation_is_false.append(1)
            
        print(f'cur_prompt: {cur_prompt}')
        print(f'label: {label}, pred: {pred}')
        print(f'validation_is_false[-1]: {validation_is_false[-1]}')
        
    ans_file.close()
    
    
    ### 计算 AUROC
    print(f'validation_is_false: {validation_is_false}')
    
    auroc_regular_entropy = calculate_auroc(validation_is_false, all_regular_entropy)
    print(f'all_regular_entropy: {all_regular_entropy}')
    print(f'auroc of regular_entropy: {auroc_regular_entropy}')
    
    auroc_regular_entropy_rao = calculate_auroc(validation_is_false, all_regular_entropy_rao)
    print(f'all_regular_entropy_rao: {all_regular_entropy_rao}')
    print(f'auroc of regular_entropy_rao: {auroc_regular_entropy_rao}')
    
    auroc_semantic_entropy = calculate_auroc(validation_is_false, all_semantic_entropy)
    print(f'all_semantic_entropy: {all_semantic_entropy}')
    print(f'auroc of semantic_entropy: {auroc_semantic_entropy}')
    
    auroc_semantic_entropy_rao = calculate_auroc(validation_is_false, all_semantic_entropy_rao)
    print(f'all_semantic_entropy_rao: {all_semantic_entropy_rao}')
    print(f'auroc of semantic_entropy_rao: {auroc_semantic_entropy_rao}')
    
    auroc_cluster_assignment_entropy = calculate_auroc(validation_is_false, all_cluster_assignment_entropy)
    print(f'all_cluster_assignment_entropy: {all_cluster_assignment_entropy}')
    print(f'auroc of cluster_assignment_entropy: {auroc_cluster_assignment_entropy}')
    
    
    ### 计算 AURAC
    aurac_regular_entropy = calculate_aurac(all_regular_entropy, validation_is_false)
    print(f'aurac of regular_entropy: {aurac_regular_entropy}')
    aurac_regular_entropy_rao = calculate_aurac(all_regular_entropy_rao, validation_is_false)
    print(f'aurac of regular_entropy_rao: {aurac_regular_entropy_rao}')
    aurac_semantic_entropy = calculate_aurac(all_semantic_entropy, validation_is_false)
    print(f'aurac of semantic_entropy: {aurac_semantic_entropy}')
    aurac_semantic_entropy_rao = calculate_aurac(all_semantic_entropy_rao, validation_is_false)
    print(f'aurac of semantic_entropy_rao: {aurac_semantic_entropy_rao}')
    aurac_cluster_assignment_entropy = calculate_aurac(all_cluster_assignment_entropy, validation_is_false)
    print(f'aurac of cluster_assignment_entropy: {aurac_cluster_assignment_entropy}')

    
    ### 二次检测
    print("----------------------二次检测-----------------------------")
    print(f"all_cluster_assignment_entropy: {all_cluster_assignment_entropy}; labels: {labels}")
    
    new_check_is_false, new_entropy_list = check_again(data_loader_check, questions, all_cluster_assignment_entropy, labels, validation_is_false, original_pred)
    
    
    ### 重新计算AURAC
    aurac_cluster_assignment_entropy_check = calculate_aurac(new_entropy_list, new_check_is_false)
    print("-------------------------------------------------------------")
    print(f"new_entropy_list: {new_entropy_list}")
    print(f"validation_is_false: {validation_is_false}")
    print(f"new_check_is_false: {new_check_is_false}")
    
    
    print(f'aurac of cluster_assignment_entropy: {aurac_cluster_assignment_entropy}')
    print(f'aurac of semantic_entropy_rao_check: {aurac_cluster_assignment_entropy_check}')
    
    # 执行比较并获取结果
    diff, both_zero, both_one, zero_one, one_zero, origin_acc, revised_acc = compare_lists(validation_is_false, new_check_is_false)

    # 输出差异结果
    if diff:
        for d in diff:
            print(f"位置下标：{d[0]}, list1的元素：{d[1]}, list2的元素：{d[2]}")
    else:
        print("两个列表完全相同")

    # 输出四类统计结果
    print("\n统计结果：")
    print(f"均为0的数量：{both_zero}")
    print(f"均为1的数量：{both_one}")
    print(f"list1为0且list2为1的数量：{zero_one}")
    print(f"list1为1且list2为0的数量：{one_zero}")
    print(f"原始准确率：{origin_acc}")
    print(f"修正后准确率：{revised_acc}")
    
    
    
    
    
    
    ### 保存实验数据    
    # 保存以上变量到pkl
    calc_data = {
        'all_question_ids': all_question_ids,
        'all_multi_responses': all_multi_responses,
        'all_multi_sequences': all_multi_sequences,
        'all_multi_log_liks': all_multi_log_liks,
        'all_log_liks_agg': all_log_liks_agg,
        'all_semantic_ids': all_semantic_ids,
        'all_log_likelihood_per_semantic_id': all_log_likelihood_per_semantic_id
    }
    calc_data_path = os.path.join(args.pkl_folder, 'calc_values.pkl')
    print(f"Saving calculation data to {calc_data_path}")
    with open(calc_data_path, 'wb') as f:
        pickle.dump(calc_data, f)
    
    # 保存熵值到pkl
    entropy_data = {
        'validation_is_false': validation_is_false,
        'all_regular_entropy': all_regular_entropy,
        'all_regular_entropy_rao': all_regular_entropy_rao,
        'all_semantic_entropy': all_semantic_entropy,
        'all_semantic_entropy_rao': all_semantic_entropy_rao,
        'all_cluster_assignment_entropy': all_cluster_assignment_entropy,
        'auroc_regular_entropy': auroc_regular_entropy,
        'auroc_regular_entropy_rao': auroc_regular_entropy_rao,
        'auroc_semantic_entropy': auroc_semantic_entropy,
        'auroc_semantic_entropy_rao': auroc_semantic_entropy_rao,
        'auroc_cluster_assignment_entropy': auroc_cluster_assignment_entropy,
        'aurac_regular_entropy': aurac_regular_entropy,
        'aurac_regular_entropy_rao': aurac_regular_entropy_rao,
        'aurac_semantic_entropy': aurac_semantic_entropy,
        'aurac_semantic_entropy_rao': aurac_semantic_entropy_rao,
        'aurac_cluster_assignment_entropy': aurac_cluster_assignment_entropy
    }
    entropy_path = os.path.join(args.pkl_folder, 'entropy_values.pkl')
    with open(entropy_path, 'wb') as f:
        pickle.dump(entropy_data, f)
        
    check_data = {
        'new_check_is_false': new_check_is_false,
        'new_entropy_list': new_entropy_list,
        'diff': diff,
        'both_zero': both_zero,
        'both_one': both_one,
        'zero_one': zero_one,
        'one_zero': one_zero,
        'origin_acc': origin_acc,
        'revised_acc': revised_acc
    }
    
    
    # 保存输出结果到JSON
    with open(json_output_path, "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)

    print(f"Results saved to:\n- {answers_file}\n- {json_output_path}\n")
    
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--annotation-dir", type=str, default="annotations")
    parser.add_argument("--greedy-search-results-file", type=str, default="greedy_search_results.jsonl")
    parser.add_argument("--pkl-folder", type=str, default="./playground/data/eval/pope/answers")
    args = parser.parse_args()

    eval_model(args)

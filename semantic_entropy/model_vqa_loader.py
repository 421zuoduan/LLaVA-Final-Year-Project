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

from typing import Tuple, Optional
from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeepSeek
from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import UncertaintyMeasures

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def compute_transition_scores(
    self,
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
    if normalize_logits:
        scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
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
    beam_sequence_indices = beam_indices * self.config.vocab_size

    # 7. Define which indices contributed to scores
    cut_idx = sequences.shape[-1] - max_beam_length
    indices = sequences[:, cut_idx:] + beam_sequence_indices

    # 8. Compute scores
    transition_scores = scores.gather(0, indices)

    # 9. Mask out transition_scores of beams that stopped early
    transition_scores[beam_indices_mask] = 0

    return transition_scores


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

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

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
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
    
    # 准备保存详细结果的字典和文件路径
    json_output_path = os.path.join(os.path.dirname(answers_file), "detailed_results.json")
    generations = {}  # 存储所有问题的详细生成结果
    
    # 准备 Entailment Model 和 UncertaintyComputer
    # entailment_model = EntailmentDeepSeek(entailment_cache_id=None, entailment_cache_only=False)

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
                    use_cache=True)
                
            n_generated = len(output_ids.scores)
            
            # 保存所有生成的序列
            all_sequences = []
            # 保存生成的自然语言文本
            all_responses = []
            # 保存生成的log likelihood
            all_log_liks = []
            # 保存不确定性指标
            uncertainty_measures = {}
            
            all_sequences.append(output_ids.sequences)
            # all_responses.append(tokenizer.decode(output_ids.sequences[0, input_ids.shape[1]:]).strip())
            # 下面这行代码用的是 model_vqa_loader.py 源代码 batch_decode 方法, 不知道与 decode 方法有什么区别
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            all_responses.append(output_text)
            
            # compute_transition_scores 得到生成序列所有 token 的 logit, 存到 log_likelihoods 里
            # 返回的 transition_scores 形状为 (batch_size*num_return_sequences, sequence_length)
            # 每个元素表示对应位置 token 的生成分数。可通过求和（应用长度惩罚）得到序列总分，与 model.generate() 返回的 sequences_scores 一致
            transition_scores = compute_transition_scores(output_ids.sequences, output_ids.scores, normalize_logits=True)
            log_likelihoods = [score.item() for score in transition_scores[0]]
            if len(log_likelihoods) == 1:
                # logging.warning('Taking first and only generation for log likelihood!')
                log_likelihoods = log_likelihoods
            else:
                log_likelihoods = log_likelihoods[:n_generated]
                
            all_log_liks.append(torch.tensor(log_likelihoods))
            
            
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
    ans_file.close()
    
    log_liks_agg = torch.tensor([torch.mean(log_lik) for log_lik in all_log_liks])
    
    ### Compute naive entropy.
    regular_entropy = uncertainty_computer.predictive_entropy(log_liks_agg)
    regular_entropy_rao = uncertainty_computer.predictive_entropy_rao(log_liks_agg)
    print(f'regular_entropy: {regular_entropy}')
    print(f'regular_entropy_rao: {regular_entropy_rao}')
    
    # ### Compute semantic entropy
    # semantic_ids = uncertainty_computer.get_semantic_ids(
    #                 strings_list=all_responses, model=entailment_model,
    #                 strict_entailment=True, example=uncertainty_computer.question)
    # print(f'semantic_ids: {semantic_ids}')
    # # Compute entropy from frequencies of cluster assignments, namely DSE
    # cluster_assignment_entropy=uncertainty_computer.cluster_assignment_entropy(semantic_ids)
    # # Compute semantic entropy.
    # log_likelihood_per_semantic_id = uncertainty_computer.logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
    # print(f'log_likelihood_per_semantic_id: {log_likelihood_per_semantic_id}')
    # pe = uncertainty_computer.predictive_entropy_rao(torch.tensor(log_likelihood_per_semantic_id))
    # # entropies['semantic_entropy'].append(pe)
    # semantic_entropy = pe
    # print(f'cluster_assignment_entropy: {cluster_assignment_entropy}')
    # print(f'semantic_entropy: {semantic_entropy}')
    
    
    # 保存详细结果到JSON
    with open(json_output_path, "w") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)

    print(f"Results saved to:\n- {answers_file}\n- {json_output_path}\n- {pkl_output_path}")
    

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
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    eval_model(args)

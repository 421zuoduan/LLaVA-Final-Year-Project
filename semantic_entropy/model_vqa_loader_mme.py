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

from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeepSeek
from semantic_entropy.uncertainty.uncertainty_measures.semantic_entropy import UncertaintyMeasures
from semantic_entropy.uncertainty.utils.doubao import predict_doubao
from semantic_entropy.uncertainty.utils.siliconflow import predict_qwen


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


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

    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)

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

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

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
    args = parser.parse_args()

    eval_model(args)

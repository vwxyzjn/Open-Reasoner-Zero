"""
This snippet shows how packing samples work from Open-Reasoner-Zero.

"""
from typing import List, Optional

import torch
from transformers import AutoTokenizer


def _tokenize(tokenizer, texts, max_length=99999999, padding=True, device=None):
    if not padding:
        # when padding is False, return tokenized texts as list
        return tokenizer(
            texts,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
    batch = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_length,
        padding=True,
        truncation=True,
    )
    return {k: v.to(device) for k, v in batch.items()}


def _convert_prompts_outputs_to_batch_tensors_packing(
    prompts: List[str], outputs: List[str], custom_rewards: Optional[List[torch.Tensor]], packing_max_len: int, tokenizer: int,
    prompt_max_len: int, generate_max_len: int,
):
    ret_sequences = []
    ret_attention_masks = []
    ret_num_actions = []
    ret_packed_seq_lens = []
    if custom_rewards is not None:
        ret_custom_rewards = []
    else:
        ret_custom_rewards = None

    assert (
        len(prompts) == len(outputs) and len(prompts) > 0
    ), "prompts and outputs must have the same length and length must be greater than 0"

    def _new_instance():
        out_sequence = torch.full((packing_max_len,), torch.tensor(tokenizer.pad_token_id), dtype=torch.long)
        out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
        out_num_actions = []
        out_packed_seq_lens = []
        rewards = [] if custom_rewards else None
        seq_offset = 0
        seq_index = 0
        return (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
        )

    def _accumulate(
        out_sequence,
        out_attention_mask,
        out_num_actions,
        out_packed_seq_lens,
        rewards,
        seq_offset,
        seq_index,
        sequence,
        attention_mask,
        num_action,
        total_len,
        custom_rewards,
        i,
    ):
        out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
        out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
        out_num_actions.append(num_action)
        out_packed_seq_lens.append(total_len)
        if custom_rewards:
            rewards.append(custom_rewards[i])
        return seq_offset + total_len, seq_index + 1

    sequences = []
    attention_masks = []
    num_actions = []
    total_lens = []

    input_token_ids = _tokenize(tokenizer, prompts, prompt_max_len, padding=False)["input_ids"]
    response_token_ids = _tokenize(tokenizer, outputs, generate_max_len, padding=False)["input_ids"]

    for input_ids, response_ids in zip(input_token_ids, response_token_ids):
        sequences.append(input_ids + response_ids)
        attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
        num_actions.append(len(response_ids))
        total_lens.append(len(input_ids) + len(response_ids))

    # make packed sequences
    (
        out_sequence,
        out_attention_mask,
        out_num_actions,
        out_packed_seq_lens,
        rewards,
        seq_offset,
        seq_index,
    ) = _new_instance()
    for i, (sequence, attention_mask, num_action, total_len) in enumerate(
        zip(sequences, attention_masks, num_actions, total_lens)
    ):
        if seq_offset + total_len < packing_max_len:
            seq_offset, seq_index = _accumulate(
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
                sequence,
                attention_mask,
                num_action,
                total_len,
                custom_rewards,
                i,
            )
        elif seq_offset + total_len == packing_max_len:
            seq_offset, seq_index = _accumulate(
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
                sequence,
                attention_mask,
                num_action,
                total_len,
                custom_rewards,
                i,
            )
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)
            (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
            ) = _new_instance()
        elif seq_offset + total_len > packing_max_len:
            if seq_offset > 0:
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                ) = _new_instance()
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )

    if seq_offset > 0:
        valid_size = out_attention_mask.nonzero().size(0)
        ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
        ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
        ret_num_actions.append(out_num_actions)
        ret_packed_seq_lens.append(out_packed_seq_lens)
        if custom_rewards:
            ret_custom_rewards.append(rewards)

    return ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    prompts = ["User: Hello, how are you?\nAssistant: <think>", "User: What is the capital of France?\nAssistant: <think>", "User: What is the capital of Germany?\nAssistant: <think>"]
    outputs = ["I'm good, thank you!", "Paris", "Berlin"]
    custom_rewards = [1.0] * len(prompts)
    ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards = _convert_prompts_outputs_to_batch_tensors_packing(
        prompts=prompts, 
        outputs=outputs, 
        custom_rewards=custom_rewards, 
        packing_max_len=40, 
        tokenizer=tokenizer, 
        prompt_max_len=20, 
        generate_max_len=20,
    )
    # print("there are %d sequences" % len(ret_sequences))
    print(f"{len(prompts)} sequences are packed into {len(ret_sequences)} sequences")
    print(repr(tokenizer.decode(ret_sequences[0][0])))
    print(repr(tokenizer.decode(ret_sequences[1][0])))
    print(ret_sequences[0])
    print(ret_attention_masks[0])
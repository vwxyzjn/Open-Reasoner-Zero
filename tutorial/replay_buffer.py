from dataclasses import dataclass
from functools import partial
import random
import time
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from orz.ppo.models import get_llm_for_sequence_regression
from orz.ppo.replay_buffer import NaiveReplayBuffer
from packing import _convert_prompts_outputs_to_batch_tensors_packing, _tokenize
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from rich.pretty import pprint

from inference import inference_and_calculates, Actor, Experience
from adv_and_returns import _calc_advantages_and_returns

class Timer:
    """A context manager for timing code blocks"""

    def __init__(self, description: str, noop: int = 0):
        self.description = description
        self.noop = noop

    def __enter__(self):
        if self.noop:
            return
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        if self.noop:
            return
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        print(f"{self.description}: {self.duration} seconds")

def normalize_advantages(buffer):
    items = []
    action_masks = []
    for item in buffer:
        items.append(getattr(item, "advantages"))
        action_masks.append(item.action_mask)

    items_vector = torch.cat(items).float().flatten()

    if action_masks[0] is None:
        # packing samples has no action mask
        action_masks_vector = 1
        num_actions = items_vector.numel()
    else:
        action_masks_vector = torch.cat(action_masks).flatten()
        num_actions = action_masks_vector.sum()

    # mean
    mean = items_vector.mean()
    # std
    std = ((items_vector - mean).pow(2) * action_masks_vector).sum()
    rstd = (std / num_actions).clamp(min=1e-8).rsqrt()

    for i, item in enumerate(buffer):
        t = (items[i] - mean) * rstd
        setattr(item, "advantages", t.bfloat16())
    return buffer



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = Actor(pretrain_or_model="EleutherAI/pythia-14m", packing_samples=True)
    critic_model = get_llm_for_sequence_regression("EleutherAI/pythia-14m", "critic", packing_samples=True)
    ref_model = Actor(pretrain_or_model="EleutherAI/pythia-14m", packing_samples=True)
    prompts = [
        "User: Hello, how are you?\nAssistant: <think>",
        "User: What is the capital of France?\nAssistant: <think>",
        "User: What is the capital of Germany?\nAssistant: <think>",
    ]
    outputs = [
        "I'm good, thank you!",
        "Paris",
        "Berlin",
    ]

    output_tokens = _tokenize(tokenizer, outputs, 20, padding=False)["input_ids"]
    custom_rewards = [torch.zeros(len(output_token)) for output_token in output_tokens]
    for i, output_token in enumerate(output_tokens):
        custom_rewards[i][-1] = 1.0
    with Timer("convert_prompts_outputs_to_batch_tensors_packing"): 
        ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards = _convert_prompts_outputs_to_batch_tensors_packing(
            prompts=prompts, 
            outputs=outputs, 
            custom_rewards=custom_rewards, 
            packing_max_len=40, 
            tokenizer=tokenizer, 
            prompt_max_len=20, 
            generate_max_len=20,
        )
    print(f"{len(prompts)} sequences are packed into {len(ret_sequences)} sequences")
    print(repr(tokenizer.decode(ret_sequences[0][0])))
    print(repr(tokenizer.decode(ret_sequences[1][0])))
    print(ret_sequences[0])
    print(ret_attention_masks[0])

    action_masks = None
    experiences = inference_and_calculates(
        ret_sequences,
        ret_attention_masks,
        action_masks,
        ret_num_actions,
        ret_packed_seq_lens,
        ret_custom_rewards,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        critic_num_nodes=1,
        critic_num_gpus_per_node=1,
        ref_num_nodes=1,
        ref_num_gpus_per_node=1,
        reward_num_nodes=0,
        reward_num_gpus_per_node=0,
        micro_forward_batch_size=1,
        model=model,
        critic_model=critic_model,
        ref_model=ref_model,
        reward_model=None,
    )

    micro_train_batch_size = 2
    replay_buffer = NaiveReplayBuffer(
        sample_batch_size=micro_train_batch_size,
        limit=0,
        cpu_offload=True,
        packing_samples=True,
    )

    print("-" * 100)
    for experience in experiences:
        experience, metrics = _calc_advantages_and_returns(
            experience,
            init_kl_coef=0.0,
            reward_clip_range=None,
            use_kl_loss=False,
            gamma=0.99,
            lambd=0.98,
        )
        replay_buffer.append(experience)
        pprint((experience, metrics))

    replay_buffer = normalize_advantages(replay_buffer)
    pprint(replay_buffer)

    actor_num_nodes = 1
    actor_num_gpus_per_node = 1
    critic_num_nodes = 1
    critic_num_gpus_per_node = 1
    num_policy_dp_nodes = actor_num_nodes * actor_num_gpus_per_node
    num_critic_dp_nodes = critic_num_nodes * critic_num_gpus_per_node
    policy_buffers = replay_buffer.split_to_n_batches(num_policy_dp_nodes)
    pprint(policy_buffers)


    dataloader = DataLoader(
        replay_buffer,
        batch_size=replay_buffer.sample_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        collate_fn=replay_buffer.collate_fn,
    )
    for batch in dataloader:
        pprint(batch)
        break

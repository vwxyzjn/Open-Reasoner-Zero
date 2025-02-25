from dataclasses import dataclass
from functools import partial
import random
from typing import List, Optional, Union

from orz.ppo.models import get_llm_for_sequence_regression
from packing import _convert_prompts_outputs_to_batch_tensors_packing
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
from rich.pretty import pprint

from inference import inference_and_calculates, Actor



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = Actor(pretrain_or_model="EleutherAI/pythia-14m", packing_samples=True)
    critic_model = get_llm_for_sequence_regression("EleutherAI/pythia-14m", "critic", packing_samples=True)
    ref_model = Actor(pretrain_or_model="EleutherAI/pythia-14m", packing_samples=True)
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
    pprint(experiences)
from dataclasses import dataclass
from functools import partial
import random
from typing import List, Optional, Tuple, Union

import torch
from torch.nn.utils.rnn import pad_sequence

from orz.ppo.models import get_llm_for_sequence_regression
from packing import _convert_prompts_outputs_to_batch_tensors_packing, _tokenize
from transformers import AutoTokenizer
import torch.nn.functional as F
import torch.nn as nn
from rich.pretty import pprint

from inference import inference_and_calculates, Actor, Experience

@torch.no_grad()
def compute_reward(
    r: Optional[Union[torch.Tensor, float]],
    kl_coef: float,
    kl: Union[torch.Tensor, list[torch.Tensor]],
    custom_rewards: Optional[Union[torch.Tensor, list[torch.Tensor]]] = None,
    action_mask: Optional[torch.Tensor] = None,
    num_actions: Optional[Union[int, list[int]]] = None,
    reward_clip_range: Tuple[float, float] = None,
    use_kl_loss: bool = False,
) -> Union[torch.Tensor, list[torch.Tensor]]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    if r is not None and reward_clip_range:
        r = r.clamp(min=reward_clip_range[0], max=reward_clip_range[1])

    if action_mask is not None:
        if not use_kl_loss:
            kl_reward = -kl_coef * kl
        else:
            kl_reward = torch.zeros_like(kl)
        # The following code is equivalent to:
        #
        # last_reward = torch.zeros_like(kl)
        # for i in range(last_reward.size(0)):
        #     for t in reversed(range(last_reward.size(1))):
        #         if action_mask[i][t] > 0.5:
        #             last_reward[i][t] = r[i]
        #             break
        #
        if r is not None:
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
            last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))
            reward = last_reward + kl_reward
        else:
            reward = kl_reward
        if custom_rewards is not None:
            custom_rewards_batch = pad_sequence(custom_rewards, batch_first=True, padding_value=0.0)
            reward = reward + custom_rewards_batch
    else:
        if not use_kl_loss:
            kl_reward = -kl_coef * kl
        else:
            kl_reward = torch.zeros_like(kl)
        if r is not None:
            kl_reward[:, torch.tensor(num_actions).cumsum(dim=-1) - 1] += r

        if custom_rewards is not None:
            custom_rewards = torch.cat(custom_rewards, dim=0)
            reward = kl_reward + custom_rewards.unsqueeze(0)
        else:
            reward = kl_reward

    return reward


@torch.no_grad()
def get_advantages_and_returns(
    values: Optional[torch.Tensor],
    rewards: torch.Tensor,
    action_mask: torch.Tensor,
    num_actions: Optional[torch.Tensor],
    gamma: float,
    lambd: float,
    packing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (batch_size, response_size)
    - rewards: Tensor of shape (batch_size, response_size)

    Output:
    - advantages: Tensor of shape (batch_size, response_size)
    - returns: Tensor of shape (batch_size, response_size)
    """

    # fp32 mode
    if values is not None:
        values = values.float()
    rewards = rewards.float()

    if packing:
        accum_reverse_num_actions = torch.cumsum(torch.tensor(num_actions), dim=0)
        sample_idx = len(num_actions) - 1
    else:
        accum_reverse_num_actions = None

    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)

    # Mask invalid responses
    if action_mask is not None:
        if values is not None:
            values = action_mask * values
        rewards = action_mask * rewards

    for t in reversed(range(response_length)):
        if values is not None:
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        else:
            nextvalues = 0.0
        if packing and sample_idx >= 0 and t + 1 == accum_reverse_num_actions[sample_idx]:
            sample_idx -= 1
            lastgaelam = 0
            nextvalues = 0.0
        if values is not None:
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        else:
            delta = rewards[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    if values is not None:
        returns = advantages + values
    else:
        returns = advantages
    return advantages.detach(), returns

@torch.no_grad()
def _calc_advantages_and_returns(experience: Experience, init_kl_coef: float, reward_clip_range: Tuple[float, float], use_kl_loss: bool, gamma: float, lambd: float):
    num_actions = experience.info["num_actions"]
    reward = compute_reward(
        experience.info["reward"],
        init_kl_coef,
        experience.kl,
        custom_rewards=experience.info["custom_rewards"],
        action_mask=experience.action_mask,
        num_actions=num_actions,
        reward_clip_range=reward_clip_range,
        use_kl_loss=use_kl_loss,
    )
    experience.advantages, experience.returns = get_advantages_and_returns(
        experience.values,
        reward,
        experience.action_mask,
        num_actions,
        gamma,
        lambd,
        packing=True,
    )
    return_sums = reward.sum(dim=-1)
    return_sums /= len(num_actions)
    experience.info["return"] = return_sums
    experience.kl = None

    avg_rewards = return_sums.mean().item()
    avg_kl = experience.info["kl"].mean().item()
    avg_kl_max = experience.info["kl_max"].mean().item()

    avg_response_length = experience.info["response_length"].mean().item()
    if experience.info["reward"] is not None:
        avg_orm_score = experience.info["reward"].mean().item()
    else:
        avg_orm_score = 0

    if experience.info["custom_rewards"] is not None:

        def func(x):
            return [r.sum() for r in x]

        avg_custom_rewards = torch.stack(func(experience.info["custom_rewards"])).mean().item()
        # experience.info["avg_custom_rewards"] = torch.stack(func(experience.info["custom_rewards"]))
    else:
        avg_custom_rewards = 0

    del experience.info["num_actions"]
    del experience.info["custom_rewards"]
    del experience.info["reward"]
    del experience.info["kl_max"]
    experience.to_device("cpu")

    # for replay buffer split batch
    num_packed_samples = len(num_actions)
    return_sums /= num_packed_samples
    experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
    experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

    metrics = {
        "avg_rewards": avg_rewards,
        "avg_kl": avg_kl,
        "avg_kl_max": avg_kl_max,
        "avg_response_length": avg_response_length,
        "avg_orm_score": avg_orm_score,
        "avg_custom_rewards": avg_custom_rewards,
        "avg_advantages": experience.advantages.mean().item(),
        "avg_advantages_abs": experience.advantages.abs().mean().item(),
    }

    return experience, metrics

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
        pprint((experience, metrics))
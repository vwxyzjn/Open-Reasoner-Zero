from dataclasses import dataclass
from functools import partial
import random
from typing import List, Optional, Union
import torch
from orz.ppo.models import get_llm_for_sequence_regression
from packing import _convert_prompts_outputs_to_batch_tensors_packing
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn.functional as F
import torch.nn as nn
from rich.pretty import pprint

def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device)


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory()

def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)

@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
    use_abs_kl: bool = False,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # Besides non negative, it is also unbiased and have lower variance.
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    if use_abs_kl:
        log_ratio = log_ratio.abs()

    return log_ratio


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: Optional[torch.Tensor]
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Optional[torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = to(self.base_action_log_probs, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.returns is not None:
            self.returns = to(self.returns, device)
        if self.advantages is not None:
            self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = to(self.attention_mask, device)
        if self.action_mask is not None:
            self.action_mask = to(self.action_mask, device)
        if self.num_actions is not None:
            self.num_actions = to(self.num_actions, device)
        if self.packed_seq_lens is not None:
            self.packed_seq_lens = to(self.packed_seq_lens, device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.returns is not None:
            self.returns = pin_memory(self.returns)
        if self.advantages is not None:
            self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.num_actions is not None:
            self.num_actions = self.num_actions.pin_memory()
        if self.packed_seq_lens is not None:
            self.packed_seq_lens = self.packed_seq_lens.pin_memory()
        return self

def _split_dp_batch(batch, num_dp, drop_last=False):
    # Convert batch tuple to list of lists, handling None values
    batch_lists = []
    batch_size = None
    for item in batch:
        if item is not None:
            if batch_size is None:
                batch_size = len(item)
            batch_lists.append(item)
        else:
            batch_lists.append(None)

    if drop_last:
        dp_size = batch_size // num_dp
    else:
        dp_size = (batch_size + num_dp - 1) // num_dp
    valid_size = dp_size * num_dp

    if not drop_last:
        padding_index = None
        for i in range(len(batch_lists)):
            if batch_lists[i] is not None and (
                isinstance(batch_lists[i], torch.Tensor) or isinstance(batch_lists[i], list)
            ):
                padding_size = valid_size - len(batch_lists[i])
                if padding_size > 0:
                    if padding_index is None:
                        if padding_size > len(batch_lists[i]):
                            padding_index = random.choices(range(len(batch_lists[i])), k=padding_size)
                        else:
                            padding_index = random.sample(range(len(batch_lists[i])), padding_size)
                    if isinstance(batch_lists[i], torch.Tensor):
                        batch_lists[i] = torch.cat([batch_lists[i], batch_lists[i][padding_index]], dim=0)
                    elif isinstance(batch_lists[i], list):
                        batch_lists[i] = batch_lists[i] + [batch_lists[i][j] for j in padding_index]

    for i in range(num_dp):
        # Extract micro batch for each input list
        micro_batch = []
        for batch_list in batch_lists:
            if batch_list is None:
                micro_batch.append(None)
            elif isinstance(batch_list, torch.Tensor) or isinstance(batch_list, list):
                micro_batch.append(batch_list[i * dp_size : (i + 1) * dp_size])
            else:
                micro_batch.append(batch_list)
        yield tuple(micro_batch)

def _split_and_run_micro_batch(async_fn, batch_args, micro_size):
        # Ensure batch_args is a sequence of lists with equal length
        batch_size = len(batch_args[0])
        results = []
        # Process in micro batches
        for i in range(0, batch_size, micro_size):
            # Take slice i:i+micro_size from each argument
            micro_batch_args = []
            for arg in batch_args:
                if arg is not None:
                    if not isinstance(arg, torch.Tensor) and not isinstance(arg, list):
                        micro_batch_args.append(arg)
                    elif micro_size > 1 or isinstance(arg, torch.Tensor):
                        micro_batch_args.append(arg[i : i + micro_size])
                    else:
                        micro_batch_args.append(arg[i])
                else:
                    micro_batch_args.append(None)
            results.append(async_fn(*micro_batch_args))
        return results

def inference_and_calculates(
    sequences_all: List[torch.Tensor],
    attention_mask_all: List[torch.Tensor],
    action_mask_all: Optional[List[torch.Tensor]],
    num_actions_all: Optional[List[int]],
    packed_seq_lens_all: Optional[List[int]],
    custom_rewards_all: Optional[List[torch.Tensor]],
    actor_num_nodes: int,
    actor_num_gpus_per_node: int,
    critic_num_nodes: int,
    critic_num_gpus_per_node: int,
    ref_num_nodes: int,
    ref_num_gpus_per_node: int,
    reward_num_nodes: int,
    reward_num_gpus_per_node: int,
    micro_forward_batch_size: int,
    model: torch.nn.Module,
    critic_model: Optional[torch.nn.Module] = None,
    ref_model: Optional[torch.nn.Module] = None,
    reward_model: Optional[List[torch.nn.Module]] = None,
):
    num_policy_dp_groups = actor_num_nodes * actor_num_gpus_per_node
    num_critic_dp_groups = critic_num_nodes * critic_num_gpus_per_node
    num_ref_dp_groups = ref_num_nodes * ref_num_gpus_per_node
    num_reward_dp_groups = reward_num_nodes * reward_num_gpus_per_node

    def micro_infer_model(num_dps, model, sequences, num_actions, attention_mask, packed_seq_lens):
        dp_iterator = _split_dp_batch(
            (sequences, num_actions, attention_mask, packed_seq_lens),
            num_dps,
        )
        dp_tasks = []
        for dp_rank, (
            micro_sequences,
            micro_num_actions,
            micro_attention_mask,
            micro_packed_seq_lens,
        ) in enumerate(dp_iterator):
            # model = _get_dp_group_models(dp_rank, model_type)

            def forward_fn(
                local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
            ):
                return local_model.forward(
                    sequences=fwd_sequences,
                    num_actions=fwd_num_actions,
                    attention_mask=fwd_attention_mask,
                    packed_seq_lens=fwd_packed_seq_lens,
                )

            dp_tasks.append(
                _split_and_run_micro_batch(
                    partial(forward_fn, model),
                    (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                    micro_forward_batch_size,
                )
            )
        # results = await asyncio.gather(*dp_tasks)
        # results = [task() for task in dp_tasks]
        results = sum(dp_tasks, [])
        return results

    if action_mask_all is not None:
        num_actions_all = action_mask_all.size(1)

    # # calculate critic values
    # if cfg.colocate_all and critic_model is not None:
    #     await critic_model.backload_to_gpu()

    if critic_model is not None:
        value_ref = micro_infer_model(
            num_critic_dp_groups,
            critic_model,
            sequences_all,
            num_actions_all,
            attention_mask_all,
            packed_seq_lens_all,
        )
        values = value_ref
        # if cfg.colocate_all:
        #     values = await value_ref
        #     await critic_model.offload_to_cpu()
        

    # calculate ref log probs
    base_action_log_probs_ref = micro_infer_model(
        num_ref_dp_groups, ref_model, sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
    )
    base_log_probs = base_action_log_probs_ref

    # # handle colocate critic and reward model
    # if cfg.colocate_critic_reward and not cfg.colocate_all and critic_model is not None:
    #     values = await value_ref
    #     await critic_model.async_run_method("empty_cache")

    # handle colocate actor and ref model
    base_log_probs = base_action_log_probs_ref
    # if cfg.colocate_actor_ref or cfg.colocate_all:
    #     base_log_probs = await base_action_log_probs_ref
    #     await ref_model.async_run_method("empty_cache")

    # calculate rewards
    reward_refs = []
    if reward_model:
        reward_refs.append(
            micro_infer_model(
                num_reward_dp_groups,
                reward_model,
                sequences_all,
                num_actions_all,
                attention_mask_all,
                packed_seq_lens_all,
            )
        )

    # if cfg.colocate_all:
    #     rewards = await asyncio.gather(*reward_refs)

    # calculate action log probs
    # if cfg.colocate_all:
    #     await policy_model.backload_to_gpu()

    action_log_probs_ref = micro_infer_model(
        num_policy_dp_groups,
        model,
        sequences_all,
        num_actions_all,
        attention_mask_all,
        packed_seq_lens_all,
    )
    action_log_probs = action_log_probs_ref
    # if cfg.colocate_all:
    #     action_log_probs = await action_log_probs_ref
    #     await policy_model.offload_to_cpu()

    # wait all models done
    # if not colocate_actor_ref, then need to gather base_log_probs
    # if not colocate_critic_reward and critic_model is not None, then need to gather value
    # reward_refs is always handled at last
    # if not cfg.colocate_all:
    #     if not cfg.colocate_actor_ref:
    #         if not cfg.colocate_critic_reward and critic_model is not None:
    #             results = await asyncio.gather(
    #                 value_ref, base_action_log_probs_ref, action_log_probs_ref, *reward_refs
    #             )
    #             values, base_log_probs, action_log_probs, rewards = results[0], results[1], results[2], results[3:]
    #         else:
    #             results = await asyncio.gather(base_action_log_probs_ref, action_log_probs_ref, *reward_refs)
    #             base_log_probs, action_log_probs, rewards = results[0], results[1], results[2:]
    #     else:
    #         if not cfg.colocate_critic_reward and critic_model is not None:
    #             results = await asyncio.gather(value_ref, action_log_probs_ref, *reward_refs)
    #             values, action_log_probs, rewards = results[0], results[1], results[2:]
    #         else:
    #             results = await asyncio.gather(action_log_probs_ref, *reward_refs)
    #             action_log_probs, rewards = results[0], results[1:]

    # r = torch.stack(rewards).sum(dim=0) if len(rewards) > 0 else None
    # if not cfg.colocate_all:
    #     empty_cache_tasks = [
    #         policy_model.async_run_method("empty_cache"),
    #         ref_model.async_run_method("empty_cache"),
    #     ]
    #     if critic_model:
    #         empty_cache_tasks.append(critic_model.async_run_method("empty_cache"))
    #     if reward_model:
    #         empty_cache_tasks.extend([rm.async_run_method("empty_cache") for rm in reward_model])
    #     await asyncio.gather(*empty_cache_tasks)
    r = None

    # 6. calculate kl divergence

    experiences = []
    if critic_model is not None:
        values = values[: len(sequences_all)]
    base_log_probs = base_log_probs[: len(sequences_all)]
    action_log_probs = action_log_probs[: len(sequences_all)]
    if r is not None:
        r = r[: len(sequences_all)]
    for i in range(len(action_log_probs)):
        response_length = torch.Tensor(num_actions_all[i]).unsqueeze(0)
        total_length = torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0)
        kl = compute_approx_kl(
            action_log_probs[i],
            base_log_probs[i],
            action_mask=None,
            use_kl_estimator_k3=True,
            use_abs_kl=False,
        )
        kl_max = torch.max(kl.abs(), dim=-1)[0]
        kl_mean = masked_mean(kl, None, dim=-1)
        if r is not None:
            local_reward = r[i]
        else:
            local_reward = None
        info = {
            "kl": kl_mean,
            "kl_max": kl_max,
            "reward": local_reward,
            "custom_rewards": custom_rewards_all[i] if custom_rewards_all is not None else None,
            "response_length": response_length,
            "total_length": total_length,
            "num_actions": num_actions_all[i],
        }
        experiences.append(
            Experience(
                sequences_all[i],
                action_log_probs[i],
                base_log_probs[i],
                values[i] if critic_model is not None else None,
                None,
                None,
                attention_mask_all[i],
                None,
                response_length,
                torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0),
                info,
                kl,
            )
        )
    return experiences



def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


class Actor(nn.Module):
    """
    Actor model base class.

    Args:
        model (nn.Module): Actor Model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrain_or_model,
        use_flash_attention_2=False,
        bf16=True,
        device_map=None,
        packing_samples=False,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # https://github.com/huggingface/transformers/issues/26877
            # Use `model.generate(use_cache=True)` instead.`
            self.model.config.use_cache = False

            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns action log probs"""
        if not self.packing_samples:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            position_ids = attention_mask.long().cumsum(-1) - 1
        else:
            # reset the positions for packed samples
            position_ids = reset_position_ids(attention_mask)
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)

        if num_actions is None:
            assert return_output
            return output

        log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])

        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
            action_log_probs = []
            offset = 0
            for num_action, seq_len in zip(num_actions, packed_seq_lens):
                start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                action_log_probs.append(log_probs[:, start:end])
                offset += seq_len
            action_log_probs = torch.cat(action_log_probs, dim=1)

        if return_output:
            return (action_log_probs, output)
        else:
            return action_log_probs




if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    # moneky patching
    # model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")
    # critic_model = AutoModelForSequenceClassification.from_pretrained("EleutherAI/pythia-14m")
    # ref_model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m")

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
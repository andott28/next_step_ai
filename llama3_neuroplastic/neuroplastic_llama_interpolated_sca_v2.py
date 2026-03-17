from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from .bounded_context import apply_retrieved_chunks_to_prompt
    from .neuroplastic_llama import NeuroplasticLlama
except ImportError:  # Script-mode fallback
    from bounded_context import apply_retrieved_chunks_to_prompt
    from neuroplastic_llama import NeuroplasticLlama


class FixedResidualFluencyIntegrator(nn.Module):
    def __init__(self, fluency_weight: float = 0.05, normalize_fluency: bool = True) -> None:
        super().__init__()
        if not (0.0 <= fluency_weight <= 1.0):
            raise ValueError("fluency_weight must be in [0, 1]")
        self.fluency_weight = float(fluency_weight)
        self.normalize_fluency = bool(normalize_fluency)

    def forward(self, fluency_hidden: torch.Tensor, adapted_hidden: torch.Tensor) -> torch.Tensor:
        fluency = fluency_hidden.float()
        if self.normalize_fluency:
            fluency = F.layer_norm(fluency, (fluency.shape[-1],))
        mixed = adapted_hidden.float() + (self.fluency_weight * fluency)
        return mixed.to(dtype=adapted_hidden.dtype)


class NeuroplasticLlamaInterpolatedSCAV2(nn.Module):
    def __init__(
        self,
        model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        num_tasks: int = 10,
        neuroplasticity_enabled: bool = True,
        sca_block_size: int = 32,
        sca_block_rank: int = 4,
        sca_top_k: int = 3,
        sca_sigma: float = 1.0,
        sca_refractory_steps: int = 100,
        sca_inhibition_lambda: float = 0.0,
        sca_use_cuda: bool = True,
        kv_int4_quantization: bool = True,
        kv_cpu_offload: bool = True,
        bounded_context_enabled: bool = False,
        bounded_sink_tokens: int = 8,
        bounded_local_window_tokens: int = 10_000,
        bounded_global_window_tokens: int = 128_000,
        bounded_global_start_layer: int = 84,
        bounded_global_group_id: int = 0,
        bounded_vram_budget_gib: float = 3.5,
        summary_interval_min_tokens: int = 4_000,
        summary_interval_max_tokens: int = 8_000,
        retrieval_chunk_min_tokens: int = 300,
        retrieval_chunk_max_tokens: int = 800,
        retrieval_top_k_min: int = 8,
        retrieval_top_k_max: int = 20,
        fluency_source_layer: int = 2,
        fluency_weight: float = 0.05,
        normalize_fluency: bool = True,
    ) -> None:
        super().__init__()
        self.fluency_source_layer = int(fluency_source_layer)
        self.integrator = FixedResidualFluencyIntegrator(
            fluency_weight=fluency_weight,
            normalize_fluency=normalize_fluency,
        )
        self.base_model = NeuroplasticLlama(
            model_name=model_name,
            num_tasks=num_tasks,
            neuroplasticity_enabled=neuroplasticity_enabled,
            sca_block_size=sca_block_size,
            sca_block_rank=sca_block_rank,
            sca_top_k=sca_top_k,
            sca_sigma=sca_sigma,
            sca_refractory_steps=sca_refractory_steps,
            sca_inhibition_lambda=sca_inhibition_lambda,
            sca_use_cuda=sca_use_cuda,
            kv_int4_quantization=kv_int4_quantization,
            kv_cpu_offload=kv_cpu_offload,
            bounded_context_enabled=bounded_context_enabled,
            bounded_sink_tokens=bounded_sink_tokens,
            bounded_local_window_tokens=bounded_local_window_tokens,
            bounded_global_window_tokens=bounded_global_window_tokens,
            bounded_global_start_layer=bounded_global_start_layer,
            bounded_global_group_id=bounded_global_group_id,
            bounded_vram_budget_gib=bounded_vram_budget_gib,
            summary_interval_min_tokens=summary_interval_min_tokens,
            summary_interval_max_tokens=summary_interval_max_tokens,
            retrieval_chunk_min_tokens=retrieval_chunk_min_tokens,
            retrieval_chunk_max_tokens=retrieval_chunk_max_tokens,
            retrieval_top_k_min=retrieval_top_k_min,
            retrieval_top_k_max=retrieval_top_k_max,
        )

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    @property
    def hidden_size(self) -> int:
        return self.base_model.hidden_size

    @property
    def model(self):
        return self.base_model.model

    def set_kv_cache_mode(self, int4_quantization: Optional[bool] = None, cpu_offload: Optional[bool] = None) -> None:
        self.base_model.set_kv_cache_mode(int4_quantization=int4_quantization, cpu_offload=cpu_offload)

    def set_bounded_context_mode(
        self,
        enabled: Optional[bool] = None,
        sink_tokens: Optional[int] = None,
        local_window_tokens: Optional[int] = None,
        global_window_tokens: Optional[int] = None,
        global_start_layer: Optional[int] = None,
        global_group_id: Optional[int] = None,
    ) -> None:
        self.base_model.set_bounded_context_mode(
            enabled=enabled,
            sink_tokens=sink_tokens,
            local_window_tokens=local_window_tokens,
            global_window_tokens=global_window_tokens,
            global_start_layer=global_start_layer,
            global_group_id=global_group_id,
        )

    def build_repo_memory_index(self, root_dir: str) -> int:
        return int(self.base_model.build_repo_memory_index(root_dir))

    def retrieve_repo_chunks(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.base_model.retrieve_repo_chunks(query=query, top_k=top_k)

    def _resolve_fluency_index(self, hidden_states: List[torch.Tensor]) -> int:
        if not hidden_states:
            raise ValueError("Expected hidden states from model output")
        return max(0, min(self.fluency_source_layer, len(hidden_states) - 1))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        task_id: int = 0,
    ):
        if not self.base_model.neuroplasticity_enabled:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task_id=task_id,
            )

        prev_enabled = self.base_model.neuroplasticity_enabled
        self.base_model.neuroplasticity_enabled = False
        with torch.no_grad():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                task_id=task_id,
            )

        self.base_model.neuroplasticity_enabled = True
        adapted_out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            task_id=task_id,
        )
        self.base_model.neuroplasticity_enabled = prev_enabled

        fluency_hidden = base_out.hidden_states[self._resolve_fluency_index(base_out.hidden_states)]
        adapted_hidden = adapted_out.hidden_states[-1]
        mixed_hidden = self.integrator(fluency_hidden, adapted_hidden)
        lm_dtype = self.base_model.model.lm_head.weight.dtype
        logits = self.base_model.model.lm_head(mixed_hidden.to(dtype=lm_dtype))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        extra_hidden_states = None
        if output_hidden_states:
            extra_hidden_states = (fluency_hidden, adapted_hidden, mixed_hidden)

        if return_dict is False:
            return (loss, logits, extra_hidden_states)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=extra_hidden_states,
            attentions=adapted_out.attentions if output_attentions else None,
            past_key_values=adapted_out.past_key_values,
        )

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: float = 1.0,
        use_cache: bool = True,
        eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
        pad_token_id: Optional[int] = None,
        task_id: int = 0,
        retrieved_chunk_ids: Optional[List[torch.LongTensor]] = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation.")

        max_length = kwargs.pop("max_length", None)
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise ValueError(f"Unsupported generation kwargs in NeuroplasticLlamaInterpolatedSCAV2.generate: {unknown}")
        if max_new_tokens is None and max_length is not None:
            max_new_tokens = max(int(max_length) - int(input_ids.shape[-1]), 0)

        input_ids = input_ids.to(device=self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = attention_mask.to(device=input_ids.device)
        if self.base_model._bounded_context_enabled() and retrieved_chunk_ids:
            input_ids = apply_retrieved_chunks_to_prompt(
                prompt_ids=input_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                local_window_tokens=int(self.base_model.bounded_context_config.local_window_tokens),
            )
            attention_mask = torch.ones_like(input_ids, dtype=attention_mask.dtype, device=attention_mask.device)

        if max_new_tokens is None:
            cfg_default = getattr(self.base_model.model.generation_config, "max_new_tokens", None)
            max_new_tokens = int(cfg_default) if cfg_default is not None else 50
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            return input_ids

        generation_cfg = getattr(self.base_model.model, "generation_config", None)
        eos_ids = self.base_model._normalize_eos_token_ids(eos_token_id)
        if eos_ids is None and generation_cfg is not None:
            eos_ids = self.base_model._normalize_eos_token_ids(getattr(generation_cfg, "eos_token_id", None))

        if pad_token_id is None and generation_cfg is not None:
            cfg_pad = getattr(generation_cfg, "pad_token_id", None)
            if cfg_pad is not None:
                pad_token_id = int(cfg_pad)
        if pad_token_id is None and eos_ids:
            pad_token_id = int(eos_ids[0])

        generated = input_ids
        unfinished = torch.ones((generated.shape[0],), dtype=torch.bool, device=generated.device)
        cached_state = None

        for _ in range(max_new_tokens):
            model_inputs = generated[:, -1:] if (cached_state is not None and use_cache) else generated
            past_key_values = (
                self.base_model._restore_past_key_values(cached_state)
                if (cached_state is not None and use_cache)
                else None
            )

            outputs = self(
                input_ids=model_inputs,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                return_dict=True,
                output_hidden_states=False,
                task_id=task_id,
            )
            next_token_logits = outputs.logits[:, -1, :].float()
            next_tokens = self.base_model._sample_next_token(
                logits=next_token_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            if eos_ids is not None:
                if pad_token_id is None:
                    raise ValueError("pad_token_id is required when eos_token_id is set.")
                next_tokens = torch.where(
                    unfinished,
                    next_tokens,
                    torch.full_like(next_tokens, int(pad_token_id)),
                )

            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=-1,
            )

            if use_cache:
                if self.base_model._kv_optimized_generation_enabled():
                    cached_state = self.base_model._pack_past_key_values(outputs.past_key_values)
                else:
                    cached_state = self.base_model._to_legacy_cache(outputs.past_key_values)

            if eos_ids is not None:
                is_eos = torch.zeros_like(unfinished)
                for eos in eos_ids:
                    is_eos |= next_tokens.eq(eos)
                unfinished = unfinished & (~is_eos)
                if not unfinished.any():
                    break

        return generated

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.base_model.save_pretrained(save_directory)
        payload = {
            "architecture": "NeuroplasticLlamaInterpolatedSCAV2",
            "fluency_source_layer": self.fluency_source_layer,
            "fluency_weight": self.integrator.fluency_weight,
            "normalize_fluency": self.integrator.normalize_fluency,
        }
        with open(os.path.join(save_directory, "interpolation_config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "NeuroplasticLlamaInterpolatedSCAV2":
        cfg_path = os.path.join(pretrained_model_name_or_path, "interpolation_config.json")
        interpolation_cfg: Dict[str, Any] = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                interpolation_cfg = json.load(f)

        base_kwargs: Dict[str, Any] = {
            "neuroplasticity_enabled": kwargs.pop("neuroplasticity_enabled", True),
        }
        if "kv_int4_quantization" in kwargs:
            base_kwargs["kv_int4_quantization"] = kwargs.pop("kv_int4_quantization")
        if "kv_cpu_offload" in kwargs:
            base_kwargs["kv_cpu_offload"] = kwargs.pop("kv_cpu_offload")
        passthrough_keys = [
            "bounded_context_enabled",
            "bounded_sink_tokens",
            "bounded_local_window_tokens",
            "bounded_global_window_tokens",
            "bounded_global_start_layer",
            "bounded_global_group_id",
            "bounded_vram_budget_gib",
            "summary_interval_min_tokens",
            "summary_interval_max_tokens",
            "retrieval_chunk_min_tokens",
            "retrieval_chunk_max_tokens",
            "retrieval_top_k_min",
            "retrieval_top_k_max",
        ]
        for key in passthrough_keys:
            if key in kwargs:
                base_kwargs[key] = kwargs.pop(key)

        base_model = NeuroplasticLlama.from_pretrained(
            pretrained_model_name_or_path,
            **base_kwargs,
        )
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.fluency_source_layer = int(interpolation_cfg.get("fluency_source_layer", 2))
        wrapper.integrator = FixedResidualFluencyIntegrator(
            fluency_weight=float(interpolation_cfg.get("fluency_weight", 0.05)),
            normalize_fluency=bool(interpolation_cfg.get("normalize_fluency", True)),
        )
        wrapper.base_model = base_model
        return wrapper

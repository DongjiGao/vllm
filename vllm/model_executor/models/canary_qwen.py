# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only NVIDIA Canary-Qwen-2.5B speech recognition model.

Architecture: FastConformer encoder (NeMo) + linear projection + Qwen3-1.7B LLM.
Reference: https://huggingface.co/nvidia/canary-qwen-2.5b

Requires NeMo toolkit for the audio encoder:
    pip install nemo_toolkit[asr]
"""

from collections.abc import Iterable, Mapping
from typing import Annotated, Literal

import torch
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    MultiModalDataItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    BaseDummyInputsBuilder,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

logger = init_logger(__name__)

_AUDIO_PLACEHOLDER = "<|audioplaceholder|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END = "<|audio_end|>"
_SAMPLING_RATE = 16000


def _ensure_special_tokens(tokenizer):
    special = [_AUDIO_PLACEHOLDER, _AUDIO_START, _AUDIO_END]
    existing = set(tokenizer.get_vocab().keys())
    to_add = [t for t in special if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
_MAX_AUDIO_DURATION_S = 40.0


def _load_nemo_perception(perception_cfg: dict, output_dim: int) -> nn.Module:
    try:
        from nemo.collections.speechlm2.modules import AudioPerceptionModule
        from omegaconf import DictConfig
    except ImportError as e:
        raise ImportError(
            "NeMo is required for Canary-Qwen audio encoder. "
            "Install with: pip install nemo_toolkit[asr]"
        ) from e

    cfg = DictConfig(perception_cfg)
    if "output_dim" not in cfg:
        cfg.output_dim = output_dim
    perception = AudioPerceptionModule(cfg)
    perception.eval()
    return perception


class CanaryQwenAudioInputs(TensorSchema):
    type: Literal["audio_features"] = "audio_features"
    audio_signal: Annotated[
        torch.Tensor | list[torch.Tensor], TensorShape("b", "t")
    ]
    audio_signal_length: Annotated[torch.Tensor, TensorShape("b")]


class CanaryQwenProcessingInfo(BaseProcessingInfo):

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(
            target_sr=_SAMPLING_RATE,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"audio": 1}

    def get_max_audio_tokens(self) -> int:
        return self._estimate_audio_tokens(self.get_max_audio_len())

    def get_max_audio_len(self) -> int:
        return int(_MAX_AUDIO_DURATION_S * _SAMPLING_RATE)

    @staticmethod
    def _estimate_audio_tokens(audio_length_samples: int) -> int:
        n_fft = 512
        hop_length = 160
        stft_pad = n_fft // 2
        fbank_len = (audio_length_samples + 2 * stft_pad - n_fft) // hop_length
        kernel, stride, repeat = 3, 2, 3
        add_pad = 1 + 1 - kernel
        length = float(fbank_len)
        for _ in range(repeat):
            length = (length + add_pad) / stride + 1.0
        return max(1, int(length))


class CanaryQwenMultiModalProcessor(
    BaseMultiModalProcessor[CanaryQwenProcessingInfo],
):

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_signal=MultiModalFieldConfig.batched("audio"),
            audio_signal_length=MultiModalFieldConfig.batched("audio"),
        )

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> list[PromptUpdate]:
        def get_replacement(item_idx: int):
            audios = mm_items.get_items("audio", AudioProcessorItems)
            audio = audios.get(item_idx)
            n_tokens = self.info._estimate_audio_tokens(audio.shape[-1])
            repl_full = (
                _AUDIO_START + _AUDIO_PLACEHOLDER * n_tokens + _AUDIO_END
            )
            return PromptUpdateDetails.select_text(
                repl_full, _AUDIO_PLACEHOLDER
            )

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER,
                replacement=get_replacement,
            )
        ]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        _ensure_special_tokens(tokenizer)
        mm_data = dict(mm_data)
        audios = mm_data.pop("audios", [])

        if audios:
            import re
            audio_list = []
            audio_lengths = []
            parts = re.split(f"({re.escape(_AUDIO_PLACEHOLDER)})", prompt)
            audio_idx = 0
            for i, part in enumerate(parts):
                if part == _AUDIO_PLACEHOLDER and audio_idx < len(audios):
                    audio = audios[audio_idx]
                    audio_tensor = (
                        audio if isinstance(audio, torch.Tensor)
                        else torch.as_tensor(audio, dtype=torch.float32)
                    )
                    if audio_tensor.dim() > 1:
                        audio_tensor = audio_tensor.squeeze()
                    n_tokens = self.info._estimate_audio_tokens(
                        audio_tensor.shape[-1]
                    )
                    parts[i] = (
                        _AUDIO_START
                        + _AUDIO_PLACEHOLDER * n_tokens
                        + _AUDIO_END
                    )
                    audio_list.append(audio_tensor)
                    audio_lengths.append(audio_tensor.shape[-1])
                    audio_idx += 1

            prompt = "".join(parts)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        result = BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        if audios:
            result["audio_signal"] = audio_list
            result["audio_signal_length"] = torch.tensor(audio_lengths)
        return result


class CanaryQwenDummyInputsBuilder(
    BaseDummyInputsBuilder[CanaryQwenProcessingInfo],
):

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        **kwargs,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=self.info.get_max_audio_len(),
                num_audios=num_audios,
                overrides=audio_overrides,
            )
        }

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return "Transcribe the following: " + _AUDIO_PLACEHOLDER * num_audios


@MULTIMODAL_REGISTRY.register_processor(
    CanaryQwenMultiModalProcessor,
    info=CanaryQwenProcessingInfo,
    dummy_inputs=CanaryQwenDummyInputsBuilder,
)
class CanaryQwenForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    SupportsLoRA,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Qwen3ForCausalLM"],
            )

        llm_hidden = config.text_config.hidden_size

        with self._mark_tower_model(vllm_config, {"audio"}):
            self.perception = _load_nemo_perception(
                config.perception, output_dim=llm_hidden
            )
            self.perception = self.perception.to(torch.float32)

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def _parse_audio_input(self, **kwargs) -> CanaryQwenAudioInputs | None:
        audio_signal = kwargs.pop("audio_signal", None)
        if audio_signal is None:
            return None
        audio_signal_length = kwargs.pop("audio_signal_length", None)

        if isinstance(audio_signal, list):
            max_len = max(a.shape[-1] for a in audio_signal)
            padded = [
                torch.nn.functional.pad(a, (0, max_len - a.shape[-1]))
                for a in audio_signal
            ]
            audio_signal = torch.stack(padded, dim=0)

        if audio_signal_length is None:
            audio_signal_length = torch.tensor(
                [audio_signal.shape[-1]] * audio_signal.shape[0]
            )
        elif not isinstance(audio_signal_length, torch.Tensor):
            audio_signal_length = torch.tensor(audio_signal_length)

        return CanaryQwenAudioInputs(
            audio_signal=audio_signal,
            audio_signal_length=audio_signal_length,
        )

    def _process_audio(
        self, audio_input: CanaryQwenAudioInputs
    ) -> tuple[torch.Tensor, ...]:
        device = next(self.perception.parameters()).device
        self.perception = self.perception.to(device)

        audio_signal = audio_input.audio_signal
        if isinstance(audio_signal, list):
            audio_signal = torch.stack(audio_signal, dim=0)
        audio_signal = audio_signal.to(device=device, dtype=torch.float32)
        audio_lengths = audio_input.audio_signal_length.to(device=device)

        with torch.no_grad():
            audio_embeds, audio_embed_lens = self.perception(
                input_signal=audio_signal,
                input_signal_length=audio_lengths,
            )

        audio_embeds = audio_embeds.to(torch.bfloat16)

        logger.info("_process_audio: input_len=%s, embed_shape=%s, embed_lens=%s, dtype=%s",
                     audio_lengths.tolist(), audio_embeds.shape, audio_embed_lens.tolist(), audio_embeds.dtype)

        return tuple(
            audio_embeds[i, :audio_embed_lens[i]]
            for i in range(audio_embeds.shape[0])
        )

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is None:
            logger.warning("embed_multimodal: NO audio_signal in kwargs=%s", list(kwargs.keys()))
            return []
        result = self._process_audio(audio_input)
        logger.info("embed_multimodal: returned %d embeddings, shapes=%s",
                     len(result), [r.shape for r in result])
        return result

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        pass  # debug logging removed to avoid CUDA graph capture issues
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector="perception.proj",
            tower_model="perception.encoder",
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        llm_weights = {}
        perception_weights = {}
        embed_weights = {}
        lora_a_weights = {}
        lora_b_weights = {}

        llm_peft_prefix = "llm.base_model.model."
        perception_prefix = "perception."

        for name, tensor in weights:
            if name.startswith("embed_tokens."):
                embed_weights[name] = tensor
            elif ".lora_A." in name:
                lora_a_weights[name] = tensor
            elif ".lora_B." in name:
                lora_b_weights[name] = tensor
            elif name.startswith(perception_prefix):
                key = name[len(perception_prefix):]
                perception_weights[key] = tensor
            elif name.startswith(llm_peft_prefix):
                suffix = name[len(llm_peft_prefix):]
                if ".base_layer.weight" in suffix:
                    suffix = suffix.replace(".base_layer.weight", ".weight")
                llm_weights["language_model." + suffix] = tensor
            elif name.startswith("llm."):
                llm_weights["language_model." + name[len("llm."):]] = tensor
            else:
                llm_weights[name] = tensor

        llm_weights = self._merge_lora(llm_weights, lora_a_weights, lora_b_weights)

        float32_weights = {k: v.float() for k, v in perception_weights.items()}
        self.perception.load_state_dict(float32_weights, strict=False)
        self.perception = self.perception.to(torch.float32)
        loaded_perception = {perception_prefix + k for k in perception_weights}

        combined = []
        for name, tensor in llm_weights.items():
            combined.append((name, tensor))
        for name, tensor in embed_weights.items():
            combined.append(("language_model.model." + name, tensor))

        loader = AutoWeightsLoader(self)
        loaded_llm = loader.load_weights(iter(combined))

        return loaded_llm | loaded_perception

    def _merge_lora(
        self,
        base: dict[str, torch.Tensor],
        lora_a: dict[str, torch.Tensor],
        lora_b: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not lora_a or not lora_b:
            return base

        lora_cfg = getattr(self.config, "lora", None) or {}
        alpha = lora_cfg.get("lora_alpha", 256)
        r = lora_cfg.get("r", 128)
        scaling = alpha / r

        pairs: dict[str, dict[str, torch.Tensor]] = {}
        peft_prefix = "llm.base_model.model."

        for name, tensor in lora_a.items():
            vllm_name = "language_model." + name.replace(
                peft_prefix, ""
            ).replace(".lora_A.default.weight", ".weight")
            pairs.setdefault(vllm_name, {})["A"] = tensor

        for name, tensor in lora_b.items():
            vllm_name = "language_model." + name.replace(
                peft_prefix, ""
            ).replace(".lora_B.default.weight", ".weight")
            pairs.setdefault(vllm_name, {})["B"] = tensor

        merged = 0
        for target, pair in pairs.items():
            if "A" not in pair or "B" not in pair or target not in base:
                continue
            orig_dtype = base[target].dtype
            A = pair["A"].float()
            B = pair["B"].float()
            base[target] = (base[target].float() + scaling * B @ A).to(orig_dtype)
            merged += 1

        logger.info("Merged %d LoRA pairs (scaling=%.2f)", merged, scaling)
        return base

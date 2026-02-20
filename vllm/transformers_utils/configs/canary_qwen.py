# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Configuration for nvidia/canary-qwen-2.5b (NeMo SpeechLLM2 SALM).

The HuggingFace config.json uses NeMo format which doesn't have standard
``architectures`` or ``model_type`` fields. Users must pass hf_overrides:
    --hf-overrides '{"architectures": ["CanaryQwenForConditionalGeneration"],
                     "model_type": "canary_qwen"}'
"""

from transformers import AutoConfig, PretrainedConfig


class CanaryQwenConfig(PretrainedConfig):
    model_type = "canary_qwen"

    def __init__(
        self,
        perception: dict | None = None,
        pretrained_llm: str = "Qwen/Qwen3-1.7B",
        pretrained_asr: str = "nvidia/canary-1b-flash",
        audio_locator_tag: str = "<|audioplaceholder|>",
        lora: dict | None = None,
        prompt_format: str = "qwen",
        pretrained_weights: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.perception = perception or {}
        self.pretrained_llm = pretrained_llm
        self.pretrained_asr = pretrained_asr
        self.audio_locator_tag = audio_locator_tag
        self.lora = lora or {}
        self.prompt_format = prompt_format
        self.pretrained_weights = pretrained_weights

        self.text_config = AutoConfig.from_pretrained(pretrained_llm)
        self.text_config.architectures = ["Qwen3ForCausalLM"]

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.text_config

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for NVIDIA Canary-Qwen-2.5B speech recognition model.

Requires NeMo toolkit: pip install nemo_toolkit[asr]
"""

import pytest

from ....conftest import AudioTestAssets, VllmRunner
from ...registry import HF_EXAMPLE_MODELS

MODEL_NAME = "nvidia/canary-qwen-2.5b"
AUDIO_PLACEHOLDER = "<|audioplaceholder|>"


def _check_nemo_available():
    try:
        from nemo.collections.speechlm2.modules import AudioPerceptionModule  # noqa
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _check_nemo_available(),
    reason="NeMo toolkit is required for Canary-Qwen model. "
    "Install with: pip install nemo_toolkit[asr]",
)


@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
def test_canary_qwen_transcription(
    vllm_runner,
    audio_assets: AudioTestAssets,
    dtype: str,
    max_tokens: int,
) -> None:
    """Test basic ASR transcription with Canary-Qwen."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(MODEL_NAME)
    model_info.check_available_online(on_fail="skip")

    hf_overrides = {
        "architectures": ["CanaryQwenForConditionalGeneration"],
        "model_type": "canary_qwen",
    }

    audio, sr = audio_assets[0].audio_and_sample_rate
    assert sr == 16000, f"Expected 16kHz audio, got {sr}Hz"

    prompt = f"Transcribe the following: {AUDIO_PLACEHOLDER}"

    with vllm_runner(
        MODEL_NAME,
        dtype=dtype,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"audio": 1},
        hf_overrides=hf_overrides,
        tokenizer_name="Qwen/Qwen3-1.7B",
    ) as vllm_model:
        outputs = vllm_model.generate_greedy(
            [prompt],
            max_tokens,
            audios=[[audio]],
        )

    assert len(outputs) == 1
    _, output_str = outputs[0]
    assert len(output_str) > 0, "No text generated"
    output_lower = output_str.lower()
    assert "mary" in output_lower or "lamb" in output_lower, (
        f"Expected transcription of 'Mary had a little lamb', got: {output_str}"
    )

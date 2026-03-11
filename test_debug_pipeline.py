"""Debug: trace what happens to audio through vLLM's multimodal pipeline."""

import torch
import numpy as np
import datasets

from vllm import LLM, SamplingParams


def main():
    ds = datasets.load_dataset(
        "hf-audio/esb-datasets-test-only-sorted",
        "librispeech",
        split="test.clean",
        trust_remote_code=True,
    )

    # Patch embed_multimodal to log what it receives
    import vllm.model_executor.models.canary_qwen as cq
    _orig_embed = cq.CanaryQwenForConditionalGeneration.embed_multimodal

    def _debug_embed(self, **kwargs):
        print(f"\n[DEBUG embed_multimodal] kwargs keys: {list(kwargs.keys())}")
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
            elif isinstance(v, list):
                print(f"  {k}: list of {len(v)} items")
                if len(v) > 0 and isinstance(v[0], torch.Tensor):
                    print(f"    [0]: shape={v[0].shape}, dtype={v[0].dtype}")
            else:
                print(f"  {k}: {type(v).__name__} = {v}")
        result = _orig_embed(self, **kwargs)
        if isinstance(result, (list, tuple)):
            print(f"[DEBUG embed_multimodal] returned {len(result)} embeddings")
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"  [{i}]: shape={r.shape}, dtype={r.dtype}")
        else:
            print(f"[DEBUG embed_multimodal] returned: {type(result)}")
        return result

    cq.CanaryQwenForConditionalGeneration.embed_multimodal = _debug_embed

    # Also patch _call_hf_processor to see what it produces
    _orig_hf_proc = cq.CanaryQwenMultiModalProcessor._call_hf_processor

    def _debug_hf_proc(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        print(f"\n[DEBUG _call_hf_processor]")
        print(f"  prompt: {prompt[:100]}")
        print(f"  mm_data keys: {list(mm_data.keys()) if isinstance(mm_data, dict) else type(mm_data)}")
        if isinstance(mm_data, dict):
            for k, v in mm_data.items():
                if isinstance(v, list):
                    print(f"  {k}: list of {len(v)} items")
                    for i, item in enumerate(v[:2]):
                        if isinstance(item, np.ndarray):
                            print(f"    [{i}]: ndarray shape={item.shape}, dtype={item.dtype}")
                        elif isinstance(item, torch.Tensor):
                            print(f"    [{i}]: tensor shape={item.shape}, dtype={item.dtype}")
                        else:
                            print(f"    [{i}]: {type(item)}")
                elif isinstance(v, (np.ndarray, torch.Tensor)):
                    print(f"  {k}: shape={v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
        result = _orig_hf_proc(self, prompt, mm_data, mm_kwargs, tok_kwargs)
        print(f"[DEBUG _call_hf_processor] result keys: {list(result.keys())}")
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"  {k}: list of {len(v)}")
                if len(v) > 0 and isinstance(v[0], torch.Tensor):
                    print(f"    [0]: shape={v[0].shape}")
            else:
                print(f"  {k}: {type(v)}")
        return result

    cq.CanaryQwenMultiModalProcessor._call_hf_processor = _debug_hf_proc

    hf_overrides = {
        "architectures": ["CanaryQwenForConditionalGeneration"],
        "model_type": "canary_qwen",
    }

    print("Loading vLLM...")
    llm = LLM(
        model="nvidia/canary-qwen-2.5b",
        hf_overrides=hf_overrides,
        tokenizer="Qwen/Qwen3-1.7B",
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.4,
        limit_mm_per_prompt={"audio": 1},
    )
    sampling_params = SamplingParams(max_tokens=256, temperature=0.0)

    prompt = "<|im_start|>user\nTranscribe the following: <|audioplaceholder|><|im_end|>\n<|im_start|>assistant\n"

    # Test sample 0 and sample 9
    for idx in [0, 9]:
        item = ds[idx]
        audio_arr = item["audio"]["array"].astype(np.float32)
        sr = item["audio"]["sampling_rate"]
        ref = item.get("text", "")[:80]

        print(f"\n{'='*60}")
        print(f"SAMPLE {idx}: len={len(audio_arr)} ({len(audio_arr)/sr:.1f}s)")
        print(f"  ref: {ref}...")

        outputs = llm.generate(
            {"prompt": prompt, "multi_modal_data": {"audio": (audio_arr, sr)}},
            sampling_params,
            use_tqdm=False,
        )
        hyp = outputs[0].outputs[0].text
        print(f"  hyp: {hyp[:100]}")


if __name__ == "__main__":
    main()

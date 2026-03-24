#!/usr/bin/env python3
"""
Minimal smoke test for Ministral-3-14B-Instruct-2512.
Loads via from_pretrained and generates a short output.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    model_name = os.getenv("HF_MODEL", "mistralai/Ministral-3-14B-Instruct-2512")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = os.getenv("DEVICE_MAP") or ("cuda" if device == "cuda" else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Model: {model_name}")
    print(f"Device: {device} | device_map: {device_map} | dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    prompt = "Write a single short greeting sentence."
    inputs = tokenizer(prompt, return_tensors="pt")
    if device_map != "auto":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__":
    main()

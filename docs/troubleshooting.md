# Troubleshooting

## vLLM init can fail during CUDA graph capture

### Symptom

Some runs on the Azure A100 VM fail during vLLM engine startup with logs like:

```text
Capturing CUDA graphs ...
torch.AcceleratorError: CUDA error: operation not permitted
RuntimeError: Engine core initialization failed.
```

This happens during the vLLM warmup phase, before extraction starts.

### What this means

- This is a runtime stability issue in the `vLLM / torch / CUDA / driver` stack.
- It is not a prompt bug.
- It is not a patent extraction logic bug.
- It can be intermittent: one run may succeed, the next may fail on the same VM.

The failing step is CUDA graph capture. vLLM uses it as an optimization during startup.

### Safe workaround

Rerun with eager mode enabled:

```bash
VLLM_ENFORCE_EAGER=1 bash scripts/run_vm_real500_mistral31_vllm.sh
```

For the new package wrapper:

```bash
VLLM_ENFORCE_EAGER=1 bash scripts/run_vm_patent_extraction_vllm.sh
```

### Tradeoff

- Extraction quality should stay the same.
- Throughput may be slightly worse than the non-eager fast path.
- For strict throughput benchmarking, compare runs with the same eager setting.

### Operational guidance

- For functional validation or quality comparison, `VLLM_ENFORCE_EAGER=1` is acceptable.
- For performance measurement, prefer the non-eager path when it is stable.
- If this becomes frequent, add an automatic fallback: try normal vLLM init first, then retry with eager mode.

### Related knobs

- `VLLM_ENFORCE_EAGER=1`
- `VLLM_MAX_MODEL_LEN=16384`
- `VLLM_DOC_BATCH_SIZE=32`
- `VLLM_TOKENIZER_MODE=auto` for `Mistral-Small-3.1-24B-Instruct-2503`

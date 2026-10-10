# Qwen3-4B-Instruct-2507

This directory implements a single-request, short-context BF16 network using
PyPTO kernels on an Ascend A2/A3 device. It reads the external HuggingFace
safetensors checkpoint directly and validates its architecture before loading.

The residual width is 2560, the Q/O attention width is 4096, and the K/V width
is 1024. The network has 36 layers, 32 query heads, 8 KV heads, head dimension
128, and a 9728-wide SwiGLU intermediate. The 151936-token LM head uses the
tied embedding weights. All projection tiles cover the complete dimensions.
The LM head reads the original `[vocab, hidden]` embedding with a transposed
matmul operand. A `[hidden, vocab]` ND weight would exceed the A2/A3 ND-to-NZ
load's 65535-element row-stride limit.

`kernels.py` contains embedding, linear projections, RMSNorm, Q/K normalization,
RoPE, cache append, causal GQA, residual addition, and SwiGLU. `runner.py`
uploads weights once and retains tensors on a `ChipWorker`. CPU Torch is used
for weight layout, static positional tables, transport buffers, and greedy
selection; it does not execute the model forward pass.

The initial implementation supports one request at a time and at most 128
tokens including the generation budget. Prefill processes the prompt causally
one token at a time, using the same network and KV state as decode. Sixteen
internal rows replicate one token to satisfy cube geometry; they are not a
public batch. Each layer stores K and V separately as `[8 * 128, 128]`, ordered
by KV head, position, and head dimension. This is not the 14B paged-cache ABI.

## Running

Load an existing PyPTO environment and set all build/cache paths to the desired
workspace. The task workspace provides `scripts/qwen_env.sh` and
`scripts/run_network.sh` for this purpose. It requires `PYPTO_ENV_SCRIPT` and
`MODEL_ROOT`; it never installs or rebuilds the runtime and never saves
converted weights.

From that workspace:

```bash
bash scripts/run_network.sh --device 0 --max-new-tokens 8
```

For a partial-network diagnostic using the first three prompt tokens:

```bash
bash scripts/run_network.sh --device 0 --diagnostic-layers 1 \
  --diagnostic-token-count 3 --capture-dir "$QWEN_RUN_ROOT/single_layer"
```

Use a background launcher with a workspace-local log and PID for long runs.
Partial captures are explicitly marked `diagnostic_only`; they cannot be used
as complete model-generation evidence. Generation reports include actual
loaded paths, the source revision, and hashes of the implementation files.

## Scope

This is a correctness-oriented short-generation implementation. It does not
provide batching, paged attention, long context, quantization, or a throughput
claim. Strict full-network logits/KV parity is separate from the short-text
generation check. See the task workspace's `TASKS.md` for actual commands,
device results, failed attempts, and remaining work.

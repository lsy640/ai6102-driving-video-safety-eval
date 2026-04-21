#!/bin/bash
# Launch Qwen3.6-35B-A3B-FP8 as an OpenAI-compatible vLLM server.
# Target hardware: single NVIDIA L40s 48GB.
# All tunable parameters are managed in config.yaml at project root.

set -euo pipefail

# ── Resolve project root (two levels up from scripts/) ─────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"

# ── Read settings from config.yaml via Python ─────────────────────────────────
_cfg() {
    python3 -c "
import yaml, sys
c = yaml.safe_load(open('${PROJECT_ROOT}/config.yaml'))
keys = '$1'.split('.')
for k in keys:
    c = c[k]
print(c)
" 2>/dev/null
}

MODEL="${MODEL:-$(_cfg vllm.model)}"
PORT="${PORT:-$(_cfg vllm.port)}"
MAX_LEN="${MAX_LEN:-$(_cfg vllm.max_model_len)}"
GPU_UTIL="${GPU_UTIL:-$(_cfg vllm.gpu_memory_utilization)}"
TP_SIZE="${TP_SIZE:-$(_cfg vllm.tensor_parallel_size)}"
MAX_SEQS="${MAX_SEQS:-$(_cfg vllm.max_num_seqs)}"
MAX_BATCHED="${MAX_BATCHED:-$(_cfg vllm.max_num_batched_tokens)}"
REASONING_PARSER="${REASONING_PARSER:-$(_cfg vllm.reasoning_parser)}"
CUDA_HOME_CFG="${CUDA_HOME_CFG:-$(_cfg vllm.cuda_home)}"
KV_CACHE_DTYPE="$(_cfg vllm.kv_cache_dtype)"
ENFORCE_EAGER="$(_cfg vllm.enforce_eager)"   # true/false from yaml
PREFIX_CACHING="$(_cfg vllm.enable_prefix_caching)"

# ── Activate env_vllm conda env if not already active ─────────────────────────
if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_PREFIX##*/}" != "env_vllm" ]]; then
    if command -v module &> /dev/null; then
        module load anaconda || true
    fi
    eval "$(conda shell.bash hook)"
    conda activate env_vllm
fi

# ── TC2 CUDA paths ─────────────────────────────────────────────────────────────
export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_CFG}}"
export PATH=${CUDA_HOME}/bin:${PATH}
# Conda env libstdc++ must come BEFORE system anaconda's (needs GLIBCXX_3.4.32 for FlashInfer FP8 kernels)
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}

# ── Re-export HF credentials ───────────────────────────────────────────────────
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# ── Build optional flags ───────────────────────────────────────────────────────
EXTRA_FLAGS=""
[[ "${ENFORCE_EAGER}" == "True" || "${ENFORCE_EAGER}" == "true" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --enforce-eager"
[[ "${PREFIX_CACHING}" == "True" || "${PREFIX_CACHING}" == "true" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --enable-prefix-caching"
[[ -n "${KV_CACHE_DTYPE}" && "${KV_CACHE_DTYPE}" != "None" ]] && EXTRA_FLAGS="${EXTRA_FLAGS} --kv-cache-dtype ${KV_CACHE_DTYPE}"

echo "Serving ${MODEL} on port ${PORT}"
echo "  max_model_len=${MAX_LEN}  gpu_util=${GPU_UTIL}  tp=${TP_SIZE}  max_seqs=${MAX_SEQS}  kv_dtype=${KV_CACHE_DTYPE}"

exec vllm serve "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --max-num-seqs "${MAX_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED}" \
    --reasoning-parser "${REASONING_PARSER}" \
    ${EXTRA_FLAGS}

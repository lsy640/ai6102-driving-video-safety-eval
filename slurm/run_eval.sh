#!/bin/bash
# SLURM directives below are read from config.yaml [slurm] section.
# They must be static strings here; edit config.yaml then copy values manually.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --nodelist=TC2N08
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --mem=30G
#SBATCH --job-name=vlm_eval
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --signal=B:USR1@300

set -euo pipefail

nvidia-smi

# --- conda env ---
# Temporarily disable -u: conda activation scripts (e.g. qt-main_activate.sh)
# reference unbound variables which would abort the script under set -u.
module load anaconda
set +u
eval "$(conda shell.bash hook)"
conda activate env_vllm
set -u

# --- CUDA paths (TC2 convention) ---
export PYTHONUNBUFFERED=1
export CUDA_HOME=/apps/cuda_12.8.0
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- HF token (required for gated weights like Qwen3.6-FP8) ---
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN is not set. Use:"
    echo "  sbatch --export=ALL,HF_TOKEN=\$HF_TOKEN slurm/run_eval.sh"
    exit 1
fi
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- project paths & runtime settings from config.yaml ────────────────────────
PROJECT_ROOT="${PROJECT_ROOT:-/home/msai/lius0131/VLM}"

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

VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/$(_cfg paths.dataset)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/$(_cfg paths.results)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/$(_cfg paths.logs)}"
PORT="${PORT:-$(_cfg vllm.port)}"
NUM_SAMPLES="${NUM_SAMPLES:-$(_cfg evaluation.num_samples)}"
MODEL="${MODEL:-$(_cfg vllm.model)}"
ANNOTATION_XLSX="${ANNOTATION_XLSX:-${PROJECT_ROOT}/$(_cfg paths.annotation_xlsx)}"
# HF cache: override to scratch if home quota is tight (~35 GB for FP8 weights)
export HF_HOME="${HF_HOME:-$(_cfg paths.hf_home)}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$PROJECT_ROOT"

SCRIPT_PATH="$(scontrol show job ${SLURM_JOB_ID} | grep -oP 'Command=\K\S+')"

# --- resubmit on SLURM timeout (USR1, 5 min before walltime) ---
resubmit() {
    echo "$(date): received timeout signal, resubmitting job..."
    echo "partial results are saved to ${OUT_DIR}/.dataset.json.partial"
    ssh CCDS-TC2 "cd ${PROJECT_ROOT} && sbatch --export=ALL,HF_TOKEN=${HF_TOKEN} ${SCRIPT_PATH}" || true
    echo "$(date): resubmit issued"
}
trap 'resubmit' USR1

# ==================================================================
# stage 0: pre-download model weights (idempotent; skips if cached)
# ==================================================================
echo "[$(date +%T)] ensuring model weights are cached locally..."
hf download "$MODEL" \
    --token "${HF_TOKEN}" \
    --quiet \
    && echo "[$(date +%T)] model weights ready." \
    || { echo "ERROR: failed to download $MODEL"; exit 1; }

# ==================================================================
# stage 1: launch vLLM server in background
# ==================================================================
echo "[$(date +%T)] launching vLLM server for $MODEL on port $PORT"
MODEL="$MODEL" PORT="$PORT" \
    bash "$PROJECT_ROOT/scripts/serve_vllm.sh" \
    > "$LOG_DIR/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
echo "vLLM pid=$VLLM_PID"

cleanup() {
    echo "cleaning up vLLM (pid=$VLLM_PID)"
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
}
trap 'cleanup' EXIT

# ==================================================================
# stage 2: wait for server to be ready (up to 15 min)
# ==================================================================
echo "[$(date +%T)] waiting for server..."
for i in $(seq 1 900); do
    if curl -sf "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
        echo "server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM exited early. See $LOG_DIR/vllm_${SLURM_JOB_ID}.log"
        tail -n 50 "$LOG_DIR/vllm_${SLURM_JOB_ID}.log" || true
        exit 1
    fi
    sleep 1
done

# ==================================================================
# stage 3: run evaluation (background so trap USR1 fires)
# ==================================================================
echo "[$(date +%T)] running evaluation on $VIDEO_DIR"
python "$PROJECT_ROOT/scripts/run_evaluation.py" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUT_DIR" \
    --port "$PORT" \
    --model "$MODEL" \
    --num_samples "$NUM_SAMPLES" &
EVAL_PID=$!
wait $EVAL_PID
EXIT_CODE=$?

if [[ "${RUN_ROBUSTNESS:-0}" == "1" && ${EXIT_CODE} -eq 0 ]]; then
    echo "[$(date +%T)] running robustness controls..."
    python "$PROJECT_ROOT/scripts/run_robustness.py" \
        --video_dir "$VIDEO_DIR" \
        --output_dir "$OUT_DIR" \
        --port "$PORT" \
        --model "$MODEL" \
        --num_videos 10
fi

# ==================================================================
# stage 4: human-machine consistency analysis (no GPU needed)
# ==================================================================
if [ ${EXIT_CODE} -eq 0 ] && [ -f "$ANNOTATION_XLSX" ]; then
    echo "[$(date +%T)] computing human-machine consistency metrics..."
    python "$PROJECT_ROOT/scripts/run_compare.py" \
        --dataset_json "$OUT_DIR/dataset.json" \
        --annotation_xlsx "$ANNOTATION_XLSX" \
        --output_dir "$OUT_DIR" \
    && echo "[$(date +%T)] comparison_metrics.json written." \
    || echo "[$(date +%T)] WARNING: compare step failed (non-fatal)."
else
    if [ ! -f "$ANNOTATION_XLSX" ]; then
        echo "[$(date +%T)] SKIP compare: annotation file not found ($ANNOTATION_XLSX)"
    fi
fi

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "$(date): pipeline completed successfully."
else
    echo "$(date): evaluation exited with code ${EXIT_CODE}"
fi
exit ${EXIT_CODE}

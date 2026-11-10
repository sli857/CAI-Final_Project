#!/bin/bash
#SBATCH --job-name=convai-min
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=run-min-%j.log
#
# Minimal experiment: 1 scenario x 2 agent strategies x 2 user rules x 5 repetitions
# = 20 episodes with Llama 3.1 8B. Starts vLLM, runs run.py, exits.
# Submit from the repository folder on PACE:   sbatch pace/run_min.sbatch
# Worst case this uses 0.75 GPU-hour (the 45 minute time limit above).
# If the job is cut off, submit it again: finished episodes are skipped.

set -u
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL_CACHE="models--meta-llama--Llama-3.1-8B-Instruct"
OUT="results/min-llama3.1-8b.jsonl"
REPS=5
ENV_DIR="$HOME/scratch/vllm-env"
TEMPLATE="$HOME/scratch/vllm-templates/tool_chat_template_llama3.1_json.jinja"
PORT=$((20000 + SLURM_JOB_ID % 10000))
VLLM_LOG="run-min-${SLURM_JOB_ID}-vllm.log"

cd "$SLURM_SUBMIT_DIR"
export HF_HOME="$HOME/scratch/hf_cache"
export HF_HUB_OFFLINE=1   # use the cached model only; fail fast instead of downloading

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

SNAPSHOT_DIR=$(ls -d "$HF_HOME/hub/$MODEL_CACHE/snapshots/"* | head -n 1)
echo "node: $(hostname)   port: $PORT   date: $(date)"
echo "code commit: $(git rev-parse HEAD)"
echo "vllm: $(python -c 'import vllm; print(vllm.__version__)')"
echo "model snapshot: $(basename "$SNAPSHOT_DIR")"
echo "model generation_config.json: $(tr -d '\n' < "$SNAPSHOT_DIR/generation_config.json")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -f "$TEMPLATE" ]; then
    echo "chat template not found: $TEMPLATE"
    exit 1
fi

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser llama3_json --chat-template "$TEMPLATE" \
    --max-model-len 8192 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT   # never leave the server running

# Wait for the server (at most 15 minutes). Stop at once if vLLM dies.
READY=0
for i in $(seq 1 180); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
        READY=1
        echo "server ready after about $((i * 5)) seconds"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        break
    fi
    sleep 5
done
if [ "$READY" -ne 1 ]; then
    echo "vLLM did not start. Last lines of $VLLM_LOG:"
    tail -n 40 "$VLLM_LOG"
    exit 1
fi
echo "sampling settings reported by vLLM:"
grep -i -E "sampling param|generation config" "$VLLM_LOG" | head -n 5

python run.py --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --tasks tasks.jsonl --reps "$REPS" --out "$OUT"
STATUS=$?
echo "finished with status $STATUS at $(date)"
exit $STATUS

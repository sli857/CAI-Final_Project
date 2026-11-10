#!/bin/bash
#SBATCH --job-name=vllm-hello
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=hello-%j.log
#
# Hello world: start vLLM with one model, send two tool-calling requests, exit.
# Submit from the project folder on PACE:   sbatch pace/hello_vllm.sbatch
# Worst case this uses 0.5 GPU-hour (the 30 minute time limit above).

set -u
MODEL="meta-llama/Llama-3.1-8B-Instruct"
ENV_DIR="$HOME/scratch/vllm-env"
TEMPLATE="$HOME/scratch/vllm-templates/tool_chat_template_llama3.1_json.jinja"
PORT=$((20000 + SLURM_JOB_ID % 10000))
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VLLM_LOG="$PROJECT_DIR/hello-${SLURM_JOB_ID}-vllm.log"

export HF_HOME="$HOME/scratch/hf_cache"
export HF_HUB_OFFLINE=1   # use the cached model only; fail fast instead of downloading

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

echo "node: $(hostname)   port: $PORT   date: $(date)"
echo "vllm: $(python -c 'import vllm; print(vllm.__version__)')"
echo "model snapshot: $(ls "$HF_HOME/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots")"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TEMPLATE_ARGS=()
if [ -f "$TEMPLATE" ]; then
    TEMPLATE_ARGS=(--chat-template "$TEMPLATE")
    echo "chat template: $TEMPLATE"
else
    echo "chat template: file not found, using the model's built-in template"
fi

vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
    --enable-auto-tool-choice --tool-call-parser llama3_json "${TEMPLATE_ARGS[@]}" \
    --max-model-len 8192 --seed 0 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT   # never leave the server running

# Wait for the server (at most 15 minutes). Stop at once if vLLM dies.
for i in $(seq 1 180); do
    if curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
        echo "server ready after about $((i * 5)) seconds"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM exited during startup. Last lines of $VLLM_LOG:"
        tail -n 40 "$VLLM_LOG"
        exit 1
    fi
    sleep 5
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/health"; then
    echo "server not ready after 15 minutes. Last lines of $VLLM_LOG:"
    tail -n 40 "$VLLM_LOG"
    exit 1
fi

python "$PROJECT_DIR/pace/hello_tool_call.py" "http://127.0.0.1:$PORT" "$MODEL"
STATUS=$?
echo "finished with status $STATUS at $(date)"
exit $STATUS

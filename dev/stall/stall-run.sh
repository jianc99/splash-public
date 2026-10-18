#!/bin/bash
# stall-run.sh LABEL MEM MINUTES [MODE]: one GPU hold (the caller holds /tmp/studio-gpu.lock): serves Flash-Next Q2_0 from
# ~/dev/night/stall with --expert-memory MEM, runs stall-bench.py for MINUTES (staggered 4-lane starts with long
# prefills), samples vm_stat and iostat every second, and stops. Results: ~/dev/night/stall-res/LABEL/.
set -u
LABEL=$1 MEM=$2 MINUTES=$3
TREE=$HOME/dev/night/stall OUT=$HOME/dev/night/stall-res/$LABEL PORT=8603
MODEL=ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0 REV=ed59f92082b1e93c0e96d60a8b11aab089b52f09
DRAFT=$HOME/.cache/huggingface/hub/models--incoai-internal--Qwen3.8-Flash-Next-DFlash2/snapshots/2efd1bb50bc545b668cd842845d95360c8e4e2b5
PYDIR=$HOME/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin
export PATH=$PYDIR:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
export HF_TOKEN_PATH=/nonexistent HF_HUB_DISABLE_IMPLICIT_TOKEN=1
mkdir -p "$OUT"
log() { echo "[$(date '+%T')] $LABEL: $*"; }
cd "$TREE" || exit 2
echo "=== $LABEL $(date '+%F %T')" >> /tmp/splash-stall-diag.log
/usr/bin/caffeinate -i ./splash serve --model "$MODEL" --revision "$REV" --draft-model "$DRAFT" --language-only \
  --port "$PORT" --idle-release off --max-cache-disk 0 --expert-memory "$MEM" --ple-memory 2G > "$OUT/serve.log" 2>&1 &
SERVER=$!
(while kill -0 $SERVER 2>/dev/null; do date '+%T'; vm_stat | grep -E "Pages (free|active|inactive|wired|speculative)|Pageins|Pageouts|Swapins|Swapouts|compressor"; sleep 1; done) > "$OUT/vmstat.log" 2>&1 &
(iostat -d -w 1 > "$OUT/iostat.log" 2>&1) &
IOSTAT=$!
cleanup() {
  pkill -INT -f "(launcher\.py serve|-m server\.server) .*--port $PORT( |\$)" 2>/dev/null
  for i in $(seq 1 60); do kill -0 $SERVER 2>/dev/null || break; sleep 1; done
  pkill -KILL -f "(launcher\.py serve|-m server\.server) .*--port $PORT( |\$)" 2>/dev/null
  kill $IOSTAT 2>/dev/null
}
trap cleanup EXIT
until curl -sf --max-time 5 "http://127.0.0.1:$PORT/ready" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { log "server exited before ready"; tail -20 "$OUT/serve.log"; exit 1; }
  sleep 2
done
log "ready; running for $MINUTES min"
"$PYDIR/python3" -I "$HOME/dev/night/stall-bench.py" "$PORT" "$OUT" $((MINUTES * 60)) \
  "$HOME/dev/night/expert-store/bench/prompt_doc.md" ${4:-long} 2>&1 | tee "$OUT/bench.log"
curl -sf --max-time 30 "http://127.0.0.1:$PORT/status" > "$OUT/final-status.json" 2>/dev/null
log "done"

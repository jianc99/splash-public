#!/bin/zsh
# One GPU hold (the caller holds /tmp/m520gpu.lock) of the host-step handoff stress: idle steps at the original
# timing, idle with the runner's re-reads, the awake and 1-ms-sleep controls, then more idle stress until 9 min.
cd /tmp/hstress || exit 1
rm -f /tmp/splash-handoff-diag.log
S=./host-step-handoff-stress L=./metal-backend.metallib
start=$(date +%s)
run() { echo "--- $* $(date +%T)"; "$@" 2>&1 | grep -v "^  " | tail -4; }
run $S $L idle 1000 400
HANDOFF_REREAD=1 run $S $L idle 500 400
run $S $L awake 1000 200
run $S $L sleep 300 400
while [ $(( $(date +%s) - start )) -lt 420 ]; do run $S $L idle 1000 400; done
echo "--- diag log"; cat /tmp/splash-handoff-diag.log 2>/dev/null | head -40
echo HSTRESS_DONE $(date +%T)

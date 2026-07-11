#!/usr/bin/env bash
# 4-region scripted distilled student (Ascr): behavior-clone the non-RL scripted controller
# into ONE region-conditioned policy, matching the 4-region student context. Tests whether a
# non-RL controller generalizes when cloned. Resumable.
set -u; cd "$(dirname "$0")"
PY="/opt/miniconda3/envs/learnbath/bin/python"; LOG="scripted_student4.log"; mkdir -p d4x4
REGIONS="forearm_back forearm_front upperarm_back upperarm_front"
CTX="forearm_back,forearm_front,upperarm_back,upperarm_front"
for r in $REGIONS; do
  [ -s "d4x4/scr_${r}.npz" ] || $PY scripted_baseline.py --region "$r" --episodes 100 \
    --clear-radius 0.022 --collect "d4x4/scr_${r}.npz" --contexts "$CTX" 2>&1 | tee -a "$LOG"
done
[ -s student_scripted_4r.pt ] || $PY distill.py train --data "d4x4/scr_*.npz" --out student_scripted_4r.pt 2>&1 | tee -a "$LOG"
$PY distill.py eval --student student_scripted_4r.pt --tag Ascr --regions $REGIONS 2>&1 | tee -a "$LOG"
echo "SCRIPTED_STUDENT4_DONE" | tee -a "$LOG"

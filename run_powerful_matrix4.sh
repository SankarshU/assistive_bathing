#!/usr/bin/env bash
# run_powerful_matrix4.sh — FULL 4-region study (both arm sides x back/front), 3 seeds.
# Same proven pipeline as run_powerful_matrix.sh (self-caffeinate, net-gate, resumable,
# ABORT-never-skip), extended to all 4 regions with a CLEAN namespace (d4x4/, student_*_4r,
# 4-region distillation context) so it never collides with the 2-region run's artifacts.
#   OURS teacher = ours_pc2_<region> ; YUBIK = baseline_rl/yubik_front/yubik_upperarm_* ;
#   FLAT = flat_pc24 ; students distil the 4-region-conditioned policy.
# Trained forearm teachers are reused (checkpoints persist); only upper-arm teachers train new.
# Launch: nohup caffeinate -dimsu bash run_powerful_matrix4.sh >> powerful_matrix4.log 2>&1 & disown
set -u
cd "$(dirname "$0")"
LOG="powerful_matrix4.log"; mkdir -p d4x4
PY="/opt/miniconda3/envs/learnbath/bin/python"

if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dims -w "$$" >/dev/null 2>&1 &
  echo "[guard] caffeinate holding the Mac awake for pid $$" | tee -a "$LOG"
fi

SEEDS="1 2 3"
REGIONS="forearm_back forearm_front upperarm_back upperarm_front"
CTX="forearm_back,forearm_front,upperarm_back,upperarm_front"
MR="--episodes 200 --clear-radius 0.022 --force-lo 2 --force-hi 12"

say(){ echo -e "\n===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }
phase(){ echo -e "\n########## PHASE: $* ##########" | tee -a "$LOG"; }
bestck(){ find "trained_models/auto/$1_s$2/BEST" -name 'checkpoint-*' ! -name '*.tune_metadata' 2>/dev/null | head -1; }
have_row(){ grep -q "^$1," behavior_metrics_summary.csv 2>/dev/null; }
ours_cfg(){ case "$1" in
  forearm_back) echo ours_pc2_forearm_back ;; forearm_front) echo ours_pc2_forearm_front ;;
  upperarm_back) echo ours_pc2_upperarm_back ;; upperarm_front) echo ours_pc2_upperarm_front ;;
esac; }
yubik_cfg(){ case "$1" in
  forearm_back) echo baseline_rl ;; forearm_front) echo yubik_front ;;
  upperarm_back) echo yubik_upperarm_back ;; upperarm_front) echo yubik_upperarm_front ;;
esac; }

require_net(){
  local i
  for i in 1 2 3 4 5 6 7 8; do
    if $PY -c "import socket; s=socket.socket(); s.bind(('',0)); s.close()" >/dev/null 2>&1; then return 0; fi
    echo "[net] socket bind failing; try $i/8, wait 20s..." | tee -a "$LOG"; sleep 20
  done
  echo "[net] ABORT: network wedged. Reboot + re-run to resume." | tee -a "$LOG"; exit 3
}

teachers_ready(){ # $1=seed
  local S="$1" miss="" r
  [ -z "$(bestck flat_pc24 "$S")" ] && miss="$miss flat_pc24"
  for r in $REGIONS; do
    [ -z "$(bestck "$(ours_cfg  "$r")" "$S")" ] && miss="$miss ours:$r"
    [ -z "$(bestck "$(yubik_cfg "$r")" "$S")" ] && miss="$miss yubik:$r"
  done
  echo "$miss"
}

phase "PREFLIGHT"; require_net
echo "[preflight] network OK; caffeinate active; starting 4-region matrix." | tee -a "$LOG"

phase "NON-RL scripted baseline (4 regions)"
require_net
for r in $REGIONS; do
  if have_row "Bscr_${r}"; then echo "[skip] Bscr_${r} exists" | tee -a "$LOG"; else
    $PY scripted_baseline.py --region "$r" --seed 1 --metrics --label "Bscr_${r}" \
      --episodes 100 --clear-radius 0.022 --force-lo 2 --force-hi 12 2>&1 | tee -a "$LOG"
  fi
done

for S in $SEEDS; do
  phase "SEED $S / 1-TEACHERS (ours_pc2 x4, yubik x4, flat_pc24)"
  require_net
  $PY auto_loop.py --configs \
    ours_pc2_forearm_back ours_pc2_forearm_front ours_pc2_upperarm_back ours_pc2_upperarm_front \
    baseline_rl yubik_front yubik_upperarm_back yubik_upperarm_front flat_pc24 \
    --seed "$S" --max-chunks 6 --patience 3 2>&1 | tee -a "$LOG"

  miss="$(teachers_ready "$S")"
  if [ -n "$miss" ]; then
    echo "[seed $S] teachers missing ->$miss ; net-heal + resume once" | tee -a "$LOG"; require_net
    $PY auto_loop.py --configs \
      ours_pc2_forearm_back ours_pc2_forearm_front ours_pc2_upperarm_back ours_pc2_upperarm_front \
      baseline_rl yubik_front yubik_upperarm_back yubik_upperarm_front flat_pc24 \
      --seed "$S" --max-chunks 6 --patience 3 2>&1 | tee -a "$LOG"
    miss="$(teachers_ready "$S")"
  fi
  [ -n "$miss" ] && { echo "[seed $S] ABORT: teachers still missing ->$miss" | tee -a "$LOG"; exit 4; }
  FLAT=$(bestck flat_pc24 "$S")

  phase "SEED $S / 2-COLLECT (4 regions x ours/yubik/style0/style1)"
  require_net
  for r in $REGIONS; do
    OCK=$(bestck "$(ours_cfg "$r")" "$S"); YCK=$(bestck "$(yubik_cfg "$r")" "$S")
    [ -s "d4x4/o_s${S}_${r}.npz" ]  || $PY distill.py collect --region "$r" --checkpoint "$OCK" --contexts "$CTX"            --out "d4x4/o_s${S}_${r}.npz"  2>&1 | tee -a "$LOG"
    [ -s "d4x4/y_s${S}_${r}.npz" ]  || $PY distill.py collect --region "$r" --checkpoint "$YCK" --contexts "$CTX"            --out "d4x4/y_s${S}_${r}.npz"  2>&1 | tee -a "$LOG"
    [ -s "d4x4/so_s${S}_${r}.npz" ] || $PY distill.py collect --region "$r" --checkpoint "$OCK" --contexts "$CTX" --style 0 --out "d4x4/so_s${S}_${r}.npz" 2>&1 | tee -a "$LOG"
    [ -s "d4x4/sy_s${S}_${r}.npz" ] || $PY distill.py collect --region "$r" --checkpoint "$YCK" --contexts "$CTX" --style 1 --out "d4x4/sy_s${S}_${r}.npz" 2>&1 | tee -a "$LOG"
  done

  phase "SEED $S / 3-DISTILL (4-region students)"
  require_net
  [ -s "student_ours_4r_s${S}.pt" ]  || $PY distill.py train --data "d4x4/o_s${S}_*.npz"     --out "student_ours_4r_s${S}.pt"  2>&1 | tee -a "$LOG"
  [ -s "student_yubik_4r_s${S}.pt" ] || $PY distill.py train --data "d4x4/y_s${S}_*.npz"     --out "student_yubik_4r_s${S}.pt" 2>&1 | tee -a "$LOG"
  [ -s "student_gen_4r_s${S}.pt" ]   || $PY distill.py train --data "d4x4/[oy]_s${S}_*.npz"  --out "student_gen_4r_s${S}.pt"   2>&1 | tee -a "$LOG"
  [ -s "student_style_4r_s${S}.pt" ] || $PY distill.py train --data "d4x4/s[oy]_s${S}_*.npz" --out "student_style_4r_s${S}.pt" 2>&1 | tee -a "$LOG"

  phase "SEED $S / 4-EVAL (students + teachers + flat, all 4 regions)"
  require_net
  eval_student(){ local tag="$2" need=0 r
    for r in $REGIONS; do have_row "${tag}_${r}" || need=1; done
    [ "$need" -eq 0 ] && { echo "[skip] $tag rows exist" | tee -a "$LOG"; return; }
    if [ "${3:-__none__}" = "__none__" ]; then
      $PY distill.py eval --student "$1" --tag "$tag" --regions $REGIONS 2>&1 | tee -a "$LOG"
    else
      $PY distill.py eval --student "$1" --tag "$tag" --regions $REGIONS --style "$3" 2>&1 | tee -a "$LOG"
    fi
  }
  eval_student "student_ours_4r_s${S}.pt"  "Aours_s${S}"
  eval_student "student_yubik_4r_s${S}.pt" "Ayubik_s${S}"
  eval_student "student_gen_4r_s${S}.pt"   "Agen_s${S}"
  eval_student "student_style_4r_s${S}.pt" "AstyO_s${S}" 0
  eval_student "student_style_4r_s${S}.pt" "AstyY_s${S}" 1

  for r in $REGIONS; do
    OCK=$(bestck "$(ours_cfg "$r")" "$S"); YCK=$(bestck "$(yubik_cfg "$r")" "$S")
    have_row "Tours_s${S}_${r}"  || $PY metrics_report.py --checkpoint "$OCK"  --label "Tours_s${S}_${r}"  --region "$r" $MR 2>&1 | tee -a "$LOG"
    have_row "Tyubik_s${S}_${r}" || $PY metrics_report.py --checkpoint "$YCK"  --label "Tyubik_s${S}_${r}" --region "$r" $MR 2>&1 | tee -a "$LOG"
    have_row "Bflat_s${S}_${r}"  || $PY metrics_report.py --checkpoint "$FLAT" --label "Bflat_s${S}_${r}"  --region "$r" $MR 2>&1 | tee -a "$LOG"
  done
done

phase "AGGREGATE -> 4-region figure + table"
$PY make_icra_figure.py --seeds $SEEDS 2>&1 | tee -a "$LOG"
say "DONE (4-region). figure: ICRA_matrix_figure.png ; table: ICRA_matrix_table.txt"

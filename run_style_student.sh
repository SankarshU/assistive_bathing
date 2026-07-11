#!/usr/bin/env bash
# run_style_student.sh — STYLE-conditioned student: one network that can BE ours OR
# yubik on demand via a style bit (no blending). Collect ours teachers at style=0 and
# yubik teachers at style=1, train ONE student pi(a|obs,region,style), then eval it in
# BOTH modes and compare to the model-specific students. bash 3.2 safe.
set -u
cd "$(dirname "$0")"
LOG="style.log"; mkdir -p distill_data
say(){ echo -e "\n===== [$(date '+%F %T')] $* =====" | tee -a "$LOG"; }
bestck(){ find "trained_models/auto/$1_s1/BEST" -name 'checkpoint-*' ! -name '*.tune_metadata' 2>/dev/null | head -1; }

OFB=$(bestck abl4_endpass); OFF=$(bestck ours_front)
YFB=$(bestck baseline_rl);  YFF=$(bestck yubik_front)
for v in "$OFB" "$OFF" "$YFB" "$YFF"; do
  [ -z "$v" ] && { echo "missing a teacher -> abort" | tee -a "$LOG"; exit 1; }
done

say "COLLECT with style tags (ours=0, yubik=1), 2 regions each"
python distill.py collect --region forearm_back  --checkpoint "$OFB" --style 0 --out distill_data/sty_ours_fb.npz  2>&1 | tee -a "$LOG"
python distill.py collect --region forearm_front --checkpoint "$OFF" --style 0 --out distill_data/sty_ours_ff.npz  2>&1 | tee -a "$LOG"
python distill.py collect --region forearm_back  --checkpoint "$YFB" --style 1 --out distill_data/sty_yubik_fb.npz 2>&1 | tee -a "$LOG"
python distill.py collect --region forearm_front --checkpoint "$YFF" --style 1 --out distill_data/sty_yubik_ff.npz 2>&1 | tee -a "$LOG"

say "DISTILL one style-conditioned student (context = region + style)"
python distill.py train --data "distill_data/sty_*.npz" --out student_style.pt 2>&1 | tee -a "$LOG"

say "EVAL in BOTH modes (style 0 = ours-mode, style 1 = yubik-mode)"
python distill.py eval --student student_style.pt --style 0 --tag styl_oursmode  --regions forearm_back forearm_front 2>&1 | tee -a "$LOG"
python distill.py eval --student student_style.pt --style 1 --tag styl_yubikmode --regions forearm_back forearm_front 2>&1 | tee -a "$LOG"

python results_tracker.py 2>&1 | tee -a "$LOG"
say "DONE. styl_oursmode_* should ~= ours; styl_yubikmode_* should ~= yubik. 'Best of both' = pick better mode per region."

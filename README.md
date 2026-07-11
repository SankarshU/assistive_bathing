# One Policy for the Whole Arm — Safe Robotic Bed-Bathing

Region-conditioned policy **distillation** for assistive bed-bathing: a UR5 wipes a supine
human arm in PyBullet, and a **single** network — conditioned on the arm region plus a one-bit
**comfort dial** — replaces eight per-region RL specialists while exposing a caregiver-
selectable *gentle ↔ thorough* trade-off.

This repository is the reproducible release accompanying the ICRA draft in [`paper/`](paper/).
Every number in the paper is emitted by [`analyze_final.py`](analyze_final.py) from
[`behavior_metrics_summary.csv`](behavior_metrics_summary.csv).

---

## Paper-name ↔ code-label map

The paper uses descriptive names; the code/checkpoints use short historical labels. They mean
the same thing:

| Paper name | Code label | What it is |
|---|---|---|
| **Comfort reward** (`R_comf`)      | `ours`   | structured reward: sweep + window + end-of-pass + 2-D `(s,w)` coverage + force shaping |
| **Contact-count reward** (`R_cc`)  | `yubik`  | baseline reward: `−d_arm − 0.01‖a‖² + 5·clears` |
| **Dial student**                   | `style`  | one network distilled from all 8 teachers + a comfort-dial bit |
| **Gentle** end (dial κ=0)          | `@Ours-bit`  / `AstyO` | Comfort behavior (safer, higher in-band force) |
| **Thorough** end (dial κ=1)        | `@Yubik-bit` / `AstyY` | Contact-count behavior (more targets cleared) |
| **Flat** baseline                  | `flat` / `Bflat` | single PPO on all regions, no conditioning |
| **Scripted** (non-RL)              | `scripted` / `Bscr`,`Ascr` | geometric boustrophedon sweep |

Eval-row prefixes in the CSV: `T*` = specialist teacher, `A*` = distilled student,
`Aours/Ayubik` = single-family students, `AstyO/AstyY` = dial student at each end.

---

## Layout

```
learn_bathing/                 gym env package (registers learn_bathing:WipingEnv-v0)
  envs/wiping_env.py           env + Comfort/Contact-count rewards (flag-gated terms)
  envs/env.py                  base env, humanoid + UR5 setup, target sampling
  envs/agents/, envs/urdf/     robot + humanoid + scene assets (URDF/meshes)

learn.py                       PPO training entry (RLlib 1.13); make_env()
distill.py                     collect | train | eval | render  (Algorithm 1)
scripted_baseline.py           non-RL geometric sweep controller + BC collector
metrics_report.py              episode_metrics() — the single source of all metrics
analyze_final.py               emits every paper table (T1–T4, S1–S2) -> FINAL_RESULTS.md
make_icra_figure.py            renders ICRA_matrix_figure.png from the summary CSV
make_demos.py                  the three demo GIFs (dial / learned-vs-scripted / whole-arm)
auto_loop.py                   autonomous, resumable teacher-training driver (CONFIGS)
eval_ladder.py, aggregate_4region.py, reachability_probe.py, pose_feasibility.py,
clear_by_column.py             analysis / reachability tooling

run_powerful_matrix4.sh        FULL pipeline: teachers -> collect -> distill -> eval -> figure
run_style_student.sh           trains the dial (style) student
run_scripted_student4.sh       collects scripted rollouts -> distills scripted student

student_*_4r_s{1,2,3}.pt       distilled student weights (the reproducible policies, 5.5 MB)
behavior_metrics_summary.csv   aggregated 8-method × 4-region × 3-seed results (the source of truth)
ICRA_matrix_figure.png         main comparison figure

results/   FINAL_RESULTS.md, ICRA_draft.pdf, demos/*.gif
paper/     ICRA_draft.tex / .md / .pdf  (+ figure)
docs/      HANDOFF.md, REACHABILITY_ANALYSIS.md, RESULTS_REACHABLE.md, REVIEW_AND_NEXT_STEPS.md
```

**Not committed** (see `.gitignore`, all regenerable): `trained_models/` (~1 GB RLlib teacher
checkpoints), `d4x4/` + `distill_data/` (distillation rollout buffers), per-episode CSV dumps.

---

## Install

```bash
conda create -n learnbath python=3.8 -y
conda activate learnbath
pip install -r requirements.txt
# sanity: the env must import and register
python -c "import learn_bathing, gym; gym.make('learn_bathing:WipingEnv-v0'); print('env OK')"
```

Scripts assume they run from the repo root (the gym namespace `learn_bathing:` needs the
package importable, and scripts read `.pt`/CSV from the working directory).

---

## Reproduce

### A. Paper tables and figure (no training — seconds)

The committed `behavior_metrics_summary.csv` + student weights are enough to regenerate every
number and the figure:

```bash
python analyze_final.py --md results/FINAL_RESULTS.md   # T1–T4, S1–S2 -> FINAL_RESULTS.md
python make_icra_figure.py                              # -> ICRA_matrix_figure.png
cd paper && pdflatex ICRA_draft.tex && pdflatex ICRA_draft.tex   # -> ICRA_draft.pdf
```

### B. Re-evaluate a distilled student (needs env, no training)

```bash
# score the dial student at both ends; --regions loops all four, --tag prefixes the rows
# (each becomes <tag>_<region> in behavior_metrics_summary.csv, matching analyze_final.py)
R="forearm_back forearm_front upperarm_back upperarm_front"
python distill.py eval --student student_style_4r_s1.pt --regions $R \
  --style 0 --episodes 200 --tag AstyO_s1     # Gentle  (Comfort behavior)
python distill.py eval --student student_style_4r_s1.pt --regions $R \
  --style 1 --episodes 200 --tag AstyY_s1     # Thorough (Contact-count behavior)
```

### C. Full pipeline from scratch (retrains teachers — hours, ~1 GB)

```bash
# teachers -> collect rollouts -> distill students -> eval -> figure. Resumable
# (ABORT-never-skip; re-run to continue). Self-caffeinates so macOS won't sleep it.
nohup caffeinate -dimsu bash run_powerful_matrix4.sh >> powerful_matrix4.log 2>&1 & disown
```

### D. Demo GIFs

```bash
python make_demos.py            # results/demos/{dial, learned_vs_scripted, whole_arm}.gif
```

---

## Key result (3-seed means, 4 regions)

The comfort dial in **Thorough** mode clears **6.80 / 15** vs **5.11** for the flat multi-task
baseline (**+33 %**, 12/12 region×seed pairs, exact sign test *p* = 0.0005). Flipping the dial
to **Gentle** raises in-band (2–12 N) force adherence by **+34 %** (10/12 pairs, *p* = 0.039)
in a paired same-network test. Learning dominates the non-RL scripted sweep on every
wiping-quality metric. Full breakdown: [`results/FINAL_RESULTS.md`](results/FINAL_RESULTS.md).

## Known limitations (see `docs/REACHABILITY_ANALYSIS.md`)

One bed-facing angular column of each *back* region is physically unreachable (the humanoid
has no forearm-pronation DOF), capping targets-cleared near 10/15 there. Peak contact forces
reach 47–68 N and must be bounded before hardware. Results are simulation-only; per-seed *n* = 3.
Next experiments (pose-conditioned distillation, force-spike bounding, sim-to-real) are in
[`docs/HANDOFF.md`](docs/HANDOFF.md).

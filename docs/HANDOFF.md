# Handoff — forearm wiping study (code + results + next experiments + sim-to-real)

*Self-contained handoff so a collaborator can (1) read the finished forearm study, (2) run
the next set (pose-conditioned distillation) from a documented, tested spec, and (3) see the
path to a real setting before ICRA. Everything below is validated on the real sim unless
marked PENDING.*

---

## 1. What this experiment set is (and its status)

**Focused study: region-conditioned wiping on `forearm_back` + `forearm_front`, 3 seeds.**
Produced by one autonomous, resumable pipeline: `run_powerful_matrix.sh`.

Methods (all scored on the SAME 200-ep battery → `behavior_metrics_summary.csv`):
- **OURS specialist** = structured directional reward **+ tuned PC2 coverage term** (`ours_pc2_<region>`)
- **YUBIK specialist** = contact-counting reward (`baseline_rl`, `yubik_front`)
- **FLAT baseline** = same reward, no region conditioning (`flat_forearm`)
- **Distilled students** = region-conditioned Ours / Yubik / general / style (from `distill.py`)
- **Scripted** non-RL baseline (`scripted_baseline.py`)

**Status:** matrix RUNNING (detached, self-caffeinating, resumable). On completion it writes
`ICRA_matrix_figure.png` + `ICRA_matrix_table.txt` and all CSV rows. Live table: `python -c`
over `behavior_metrics_summary.csv`, or re-run `make_icra_figure.py --seeds 1 2 3`.

---

## 2. Key scientific findings (validated)

1. **The 3rd forearm column is physically unreachable** — it is the bed-facing underside of
   the supine limb, and the humanoid has no forearm-pronation DOF. Proven by per-target IK,
   dynamic clearability (RL + lane + raster all 0%), base-pose sweep, and pose sweep. **Full
   argument + the IK-reasoning caveat (static IK was anti-predictive) in `REACHABILITY_ANALYSIS.md`.**
   This explains the historical `task_success ≈ 0` (needs 11/15; ceiling ~10).
2. **PC2 coverage term helps** — the SVD 2nd axis is the column direction; adding an `(s,PC2)`
   coverage reward (weight tuned to **1.0** by grid search: 6.08 vs 5.68 arc-only) lifts
   within-region coverage. Term lives in `wiping_env.py` (flag `_pc2_coverage_weight`).
3. **OURS Pareto-dominates YUBIK** on the reachable arc — more coverage in *every* state cell
   (advantage map) AND ~2.4× time-in-safe-force-band. So **reward-switching / a learned gate
   is unnecessary** (we tested both — negative result worth a sentence; pre-empts "did you
   hybridise?").
4. **Reporting discipline:** report **in-band force fraction** (not mean force — Yubik's low
   force is *poking*, not gentleness); **drop `pass_completions`** from cross-method tables (it
   measures Ours' own end-of-pass gate, contaminated by the hard ends). Details in
   `RESULTS_REACHABLE.md §B`.

---

## 3. Results (relevant metrics only — the publication set)

Report these 8; everything else → appendix. (Numbers fill in as the 3-seed matrix lands;
forearm_back single-seed template in `RESULTS_REACHABLE.md §B`.)

| Metric | Better | note |
|---|---|---|
| Targets cleared /15 | ↑ | forearm_back capped ~10 (unreachable column, stated limitation) |
| In-band force fraction (2–12 N) | ↑ | **the headline safety metric** |
| Sweep consistency | ↑ | systematicity (method-neutral) |
| Clear s-coverage / t-coverage | ↑ | spatial / temporal coverage |
| 2nd-half fraction | ↑ | sustained, not front-loaded |
| Drift-term rate | ↓ | stability (slightly favours Yubik — report honestly) |
| Task success | ↑ | now meaningful (~0.3), no longer ≈0 |

**Stats for the claim:** with 3 seeds, run `aggregate_4region.py` → per-seed paired t **and**
Wilcoxon over region×seed pairs (the corrected stats). Never compare `reward_mean`.

---

## 4. Code map (what to run / edit)

| File | Role |
|---|---|
| `run_powerful_matrix.sh` | THE pipeline: teachers → collect → distill → students → eval → figure (resumable) |
| `auto_loop.py` | all training configs (`ours_pc2_*`, `baseline_rl`/`yubik_front`, `flat_forearm`, pose bins) |
| `learn_bathing/envs/wiping_env.py` | env + reward; flags `_pc2_coverage_weight`, `_cov_bins`, `_pose_range_override`, `_reachable_only` (all default-off) |
| `distill.py` | region-conditioned distillation (collect/train/eval); **pose patch in §5** |
| `metrics_report.py` | the shared metric battery (+ `--reachable-only`) |
| `make_icra_figure.py` / `aggregate_4region.py` | figure + significance |
| `REACHABILITY_ANALYSIS.md` | the reachability proof (share with the professor) |
| diagnostics | `reachability_probe.py`, `clear_by_column.py`, `config_coverage.py`, `pose_feasibility.py`, `advantage_map.py` |

---

## 5. NEXT experiment set — pose-conditioned distillation (for the collaborator)

**Goal:** make the thesis's "regions AND poses" literal — one policy `π(a | obs, region, pose)`.

**Pose bins (the new context axis) — elbow flexion (q_H index 3):**
| pose bit | name | elbow range (rad) |
|---|---|---|
| 0 | flexed | [0.40, 0.95] |
| 1 | extended | [0.95, 1.50] |
(*Bin by elbow, NOT roll j5: roll's 34° range gives near-identical bins. j4/pitch is the
optional 2nd axis. Rationale: `REACHABILITY_ANALYSIS`-style j3/j4/j5 analysis.*)

Already wired: env hook `_pose_range_override` (restricts sampling to a bin — verified) and
8 specialist configs `ours_{poseElbBent,poseElbExt}_{region}` in `auto_loop.py`.

**The one remaining code change — a documented patch to `distill.py` (apply when the env is
free, then run the smoke test below):**
1. `context_vec(region, regions, pose=None, n_poses=0)` → after the region one-hot, append a
   `n_poses`-long one-hot with index `pose` set (mirror the existing `--style` append).
2. `collect`: add `--pose <int>` + `--n-poses <int>`; set `_pose_range_override` for that bin
   in the env overrides; build cvec with the pose one-hot; store `n_poses` in the npz.
3. `train`: read `n_poses` from the npz; save it in the checkpoint dict.
4. `eval_student`: read `n_poses` from the ckpt; set `has_style = n_ctx > len(regions)+n_poses`
   (make the layout EXPLICIT — do not infer); rebuild cvec with `--pose`.
5. argparse: add `--pose` to `collect` and `eval`.

**Smoke test (MUST pass before the full pose run — env must be free):**
```bash
CK=<any forearm_back checkpoint>
python distill.py collect --region forearm_back --pose 0 --n-poses 2 \
    --contexts forearm_back,forearm_front --checkpoint $CK --out d4x/PT.npz
python distill.py train --data "d4x/PT.npz" --out student_PT.pt
python distill.py eval --student student_PT.pt --pose 0 --regions forearm_back --episodes 5
# expect: a row written, no traceback, context dim = 2 regions + 2 poses (+style if used)
```
Then a `run_pose_matrix.sh` (copy `run_powerful_matrix.sh`; teachers = the 8 `ours_pose*`
configs; add `--pose`/`--n-poses` to collect+eval; students distil across region×pose).

---

## 6. Path to a real setting (before ICRA)

The pipeline is deployment-shaped already; the sim-to-real gap items:
1. **Phase interface.** The wiping policy is Phase-4 of the Manip4Care pipeline (classical
   approach/settle → wipe → retreat). Wrap it behind a `Phase4Controller.reset(region,pose)/
   act(obs)` protocol; the scripted baseline already conforms.
2. **Observation parity.** The 90-d obs must be reproducible on hardware (joint state, EEF
   pose, tool–skin contact geometry, target/region encoding). Contact geometry is the hardest
   — needs a real force/skin-contact sensing story.
3. **Force safety.** In-band-force fraction is the metric that transfers; the **peak-force
   spikes ~55 N** seen in sim MUST be bounded before hardware (re-enable the ablated
   force-shaping rung; add a hard force clamp in the controller).
4. **Reachability on the real limb.** The underside-occlusion finding is *more* severe on a
   real arm — plan the caregiver/robot **re-positioning** step (lift/turn the limb) explicitly;
   this is where the pose-conditioned policy pays off.
5. **Scope claims:** sim-only results; cite RABBIT/Manip4Care for the sim-to-real path; do NOT
   promise hardware numbers.

---

## 7. Pending / open items (priority order)

**DONE (2026-07-10):** full 4-region × 3-seed matrix (`run_powerful_matrix4.sh`) complete;
all numbers regenerated by `analyze_final.py` → `FINAL_RESULTS.md`; **Results, abstract,
method (KL direction fix + PC2 coverage term), limitations, and conclusion rewritten in BOTH
`ICRA_draft.md` and `ICRA_draft.tex`**. Headlines now: (1) style-dial student — one network,
gentle↔thorough via one bit, consistent in 4/4 regions; (2) dial beats flat baseline +33 %,
12/12 pairs, p=0.0005; (3) paired same-network test — ours-style raises in-band force +34 %,
p=0.039; (4) method ladder — scripted < flat < specialists < students on coverage AND safety.
`pass_completions` and mean-force dropped from cross-method tables (documented reasons).

1. **Recompile `ICRA_draft.pdf`** from the updated .tex (no pdflatex on this machine —
   collaborator step).
2. **Pose-conditioned distillation** (§5) — the collaborator's next set.
3. Bound peak forces (§6.3: 47–68 N spikes) before any real-setting claim — re-enable the
   force-shaping ablation rung or add a controller-level clamp.
4. Optional polish: 4-region scripted *student* (widen `scripted_baseline.py` REGIONS_ORDER
   to 4), poke-vs-pass video pair for the presentation.

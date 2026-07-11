# Review & Next Steps

Reviewed by Fable (Claude), 2026-07-02, against `fable_review.zip` (code, README, paper
draft, results). Sections A–B are the review verdicts. Section C is the reproduction path,
D the prioritized follow-ups, E open risks, F the non-RL integration plan.

**One-line verdict: the method, code, and experimental design are sound and honest — but the
headline p-value is computed with the wrong distribution and must be fixed before anything
else happens. The claim is rescuable (details in B and D0).**

---

## A. Code review

| Area | File(s) | Verdict | Notes / issues |
|---|---|---|---|
| Env + reward | `learn_bathing/envs/wiping_env.py` | ⚠ ok with caveats | Both reward families correctly gated; ablation ladder maps cleanly to env flags; upperarm_back axis-trial-order guard present (lines ~1290, 1327–30). **Caveat 1:** Fix C (drift termination, −15) and `wipe_start_init` sit OUTSIDE the `use_baseline_reward` branch, so the Yubik baseline IS subject to them — the in-code comment "NONE of the fixes apply in this branch" and README §3.1 "byte-identical" are wrong as stated. The data confirms it matters: Tyubik_s2 rows show drift-termination rates up to 0.30. This is a *defensible experimental choice* (identical episode mechanics, only the reward differs — the right isolation) but it must be described accurately in the paper, and the misleading comments fixed. **Caveat 2:** `_env_overrides.json` is global mutable state read at env construction; eval rows do not record which overrides were active (see A-metrics row for the observed consequence). |
| Distillation | `distill.py` | ✔ ok | `_kl_diag_gauss` is the correct closed form and the correct direction — KL(teacher‖student), mode-covering. Style-bit and 4-region context handling correct; eval auto-detects style dimension. One doc mismatch: README §2.3 writes KL(student‖teacher) — code is right, README text wrong. |
| Orchestration | `auto_loop.py`, `run_icra_matrix.sh`, `run_4region_3seed.sh` | ✔ ok | Ablation ladder (`_ABL_LADDER`) is a clean cumulative design; configs for 9 methods are coherent; chunked training + best-checkpoint archiving sound. |
| Metrics / eval | `metrics_report.py`, `make_icra_figure.py`, `aggregate_4region.py` | ✖ one critical bug + one data hygiene issue | **CRITICAL: `aggregate_4region.py` line 63 computes the p-value from the normal CDF (`math.erf`), not Student's t.** With t=3.11 and n=3 seeds (df=2), the correct paired-t p ≈ 0.089, not 0.0019. The +26% headline as printed is not significant at n=3. See B/D0 for the honest rescue. **Data hygiene:** `behavior_metrics_summary.csv` contains duplicate evaluations of identical policies under different labels (`Bflat_s*` vs `flat4_s*`, `Tours_s*` vs `spec4_s*`) with IDENTICAL clears but CONTRADICTORY `terminated_drift` values (e.g. Bflat_s1_forearm_back drift=0.00 vs flat4_s1_forearm_back drift=0.18) — almost certainly two eval passes under different lingering `_env_overrides.json` states. Regenerate one canonical pass, and add the active overrides as a CSV column. |
| Baseline (non-RL) | `scripted_baseline.py` | ✔ ok | Runs through the same env/action interface and the same metric reducers — exactly right for comparability. Sensible self-documentation (validate with `--render` first). |

## B. Method / results review

- **Reward design (README §3 / paper Sec 2.4)** — sound. The 6-term teacher reward
  (two-phase distance, action, clears, sweep, window, evidence-gated end-pass) matches the
  code (`abl4_endpass`); later rungs (force band, A/D/E penalties, drift) correctly held out
  as ablation, not shipped. The "force safety emerges from structure, not force shaping"
  finding is well-supported *directionally* by the matrix (n=2 caveat already declared).
- **Distillation (Algorithm 1)** — correct BC+KL on pooled teacher rollouts with region
  one-hot (+optional style bit). Matches code. The student *beating* its specialists
  (region-avg per seed: student [6.57, 6.18, 6.52] vs specialists [5.92, 6.06, 5.72]) is
  a stronger result than the claimed "parity" — pooled-data regularization is a plausible
  mechanism and worth a sentence in the paper.
- **Claims vs `results/behavior_metrics_summary.csv`** — independently recomputed:
  - Student←Ours ≈ 6.30 ✔, Student←Yubik ≈ 6.68 ✔, scripted ≈ 4.54 ✔ (README §7 numbers reproduce).
  - **Student←Ours vs flat +26%: the effect size reproduces ✔ (6.42 vs 5.09, and the
    direction is consistent in all 3 seeds and in 11 of 12 region×seed pairs). The p=0.0019
    does NOT reproduce ✖ — it is a z-test on a t statistic (A-metrics row). Correct paired
    t at n=3 seeds: p≈0.089. Honest alternatives that ARE supported: (a) report p≈0.09 with
    n=3 and the 3/3 seed consistency; (b) sign test over the 12 region×seed pairs (11/12
    positive): p≈0.006 two-sided; (c) Wilcoxon signed-rank over the 12 pairs (~p≈0.002–0.005);
    label the unit of pairing explicitly (region×seed, seeds share training stochasticity).**
  - Student ≈ specialists (n=2 parity): supported, in fact student ≥ specialists in all
    4 regions in the n=3 table ✔.
  - Ours best on in-band force (0.177) / pass completions (0.725): consistent with the
    matrix table ✔ but n=2 — keep flagged as suggestive until D1.
- **Claims not statistically supported as currently written:** the p=0.0019 headline
  (abstract + ≥3 further places in `ICRA_draft.md/tex`); every per-method quality gap in
  the 9-method matrix (n=2, already honestly flagged in Limitations).

---

## C. Reproduce current results from scratch (do this FIRST)

1. `git clone https://github.com/SankarshU/assistive_bathing.git && cd assistive_bathing`
2. `conda env create -f environment.yml && conda activate learnbath`
3. Drop in the `code/` files from `fable_review.zip` (latest, bug-fixed versions).
4. Sanity-check Ray: `python -c "import socket;s=socket.socket();s.bind(('',0));print('OK')"`
5. Full 9-method matrix (2 seeds): `caffeinate -dimsu bash run_icra_matrix.sh >> icra_matrix.log 2>&1 &`  (~6–7 h laptop)
6. Headline (3 seeds): `caffeinate -dimsu bash run_4region_3seed.sh >> 4region_3seed.log 2>&1 &`
7. Figures: `python make_icra_figure.py --seeds 1 2` (add `--full` for the appendix).
8. Verify: Student←Ours ≈ 6.3, Student←Yubik ≈ 6.7, non-RL ≈ 4.5; student ≥ flat in every seed.
   (Do NOT verify against the printed p=0.0019 — see D0.)

## D. Next experiments (prioritized — D0 is not optional)

0. **Fix the significance test (blocks submission). STATUS: APPLIED 2026-07-02 (Fable),
   verify + finish.** Done in the repo root: (a) `aggregate_4region.py` now uses the correct
   Student-t p (scipy if present, exact closed forms otherwise) AND reports exact sign +
   Wilcoxon signed-rank tests over the region×seed pairs; rerun against the live CSV gives
   **t=3.11, p=0.0896 (df=2)** and **12/12 pairs positive, Wilcoxon p=0.0005, sign p=0.0005**
   (`ICRA_4region_table.txt` regenerated). (b) All three p=0.0019 occurrences corrected in
   BOTH `ICRA_draft.md` and `ICRA_draft.tex`, with the pairing unit stated. Remaining for
   the collaborator: **recompile `ICRA_draft.pdf` from the updated .tex** (the shipped PDF
   still shows p=0.0019), fix the same claim in `REVIEW_GUIDE.md`/`fable_review.zip` (bundle
   is now stale — re-zip from repo root), and eyeball the abstract wording. Note the bundle
   CSV differs slightly from the repo CSV (11/12 vs 12/12 positive pairs — one duplicate-eval
   discrepancy, see D2), which is itself a reason to do D2 promptly.
1. **Extend the 9-method matrix to 3 seeds** (`SEEDS="1 2 3"` in `run_icra_matrix.sh`,
   resumable) → makes in-band-force / pass-completion gaps testable (with the D0 machinery).
2. **Regenerate `behavior_metrics_summary.csv` in one canonical eval pass** (fixes the
   Bflat/flat4 + Tours/spec4 drift contradictions), and add an `overrides` column recording
   the active `_env_overrides.json` per row.
3. **Interference learning-curve figure** — flat multi-task curve vs specialists' from
   existing `run.log`s (no new training). Justifies "why distillation."
4. **Held-out-region generalization** — student distilled from 3 regions, zero-shot on the
   4th; one distill+eval run; strongest generalization claim available.
5. Documentation corrections (with D0's edit pass): README §2.3 KL direction; README §3.1
   "byte-identical" → "identical episode mechanics (incl. drift termination and wipe-start
   for all methods); only the reward differs"; same fix to the in-code comment in the
   `use_baseline_reward` branch.

## E. Open risks / caveats to close
- n=2 for most of the matrix (until D1); per-method deltas suggestive only.
- `task_success` ≈ 0 for all methods — grading on targets-cleared is the declared choice;
  keep it stated prominently (inherited 0.69 threshold is a cliff on a hard task).
- Scripted baseline "wins" isolated metrics (region-covered, spatial order) by sweeping
  wide while clearing little — interpret jointly; the professor deck already frames this.
- Drift-termination applies to ALL methods incl. Yubik and scripted (see A) — once D5's
  doc fix lands this becomes a feature (identical protocol), not a bug.
- Eval-time global state (`_env_overrides.json`) — mitigated by D2's overrides column.

## F. Integrating the two non-RL algorithms from `Project ppt.pptx`

The original deck defines the Manip4Care pipeline as a *class of methods*: **Algorithm 1
(classical approach + settle: cosine-interpolated PD to the policy-ready pose)** and
**Algorithm 2 (episode orchestration whose Phase 5 retreat is also classical)** wrap a
swappable **Phase-4 wiping controller**. Everything in this repo — specialists, students,
flat baseline, scripted sweep — is a Phase-4 candidate. That framing dictates the
integration:

1. **Interface contract (shared by all Phase-4 candidates).** Input: policy-ready state
   (post-settle joint pose, region id, warm-up steps as in Algorithm 2 Phase 4), the 90-d
   obs, and for conditioned students the context vector (region one-hot [+style]). Output:
   6-d action per step until stall/success/budget. The scripted controller already conforms
   (it runs through the same env/action interface); the student needs only the region bit
   set from the pipeline's region argument. Define this as a small `Phase4Controller`
   protocol (reset(region), act(obs)) in the demo repo and wrap all candidates.
2. **Where they slot relative to distillation.** Algorithms 1/2 are *not* alternatives to
   the student — they are the deployment harness. The two non-RL algorithms stay classical
   and unlearned (their determinism is the argument for the hybrid design, ppt slide 3).
   The scripted sweep is the non-RL Phase-4 alternative and is already benchmarked; it can
   also serve as a *distillation source* (behavior-clone its rollouts into the student with
   a style bit) — the professor deck's finding ("a non-RL controller cannot generalize even
   when cloned") is exactly this experiment and belongs in the paper as the third column of
   the class: structured-RL / counting-RL / scripted, each as specialist AND as distilled.
3. **Head-to-head benchmark under the same metrics.** Current evals score Phase-4 in
   isolation (env-reset seeding stands in for Algorithm 1). Add an *end-to-end pipeline
   eval*: run Algorithm 1 → transition → Phase-4 candidate → retreat inside
   `demo_full_pipeline_v4.py` (rl_wiping_demo repo), scoring with the SAME
   `metrics_report.episode_metrics` reducers plus two pipeline-level metrics: handoff
   success (policy-ready state reached within tolerance) and end-to-end episode time.
   This closes the sim-to-deployment gap in the paper's story and directly exercises the
   controller-transition sensitivity that standalone evals hide. Cheap: inference only,
   ~4 candidates × 4 regions × 100 episodes.
4. **Benchmark fairness rule.** All Phase-4 candidates run under identical episode
   mechanics (drift/stall termination, wipe-start, budget) — the same isolation principle
   used for the reward comparison (see A caveat 1, now documented rather than implicit).

---
*Handover note: the base repo remote is `SankarshU/assistive_bathing`. The living status
doc is `STATUS.md`; frozen history in `HANDOFF_CONTEXT_2026-06-12.md`; long-term plan in
`ICRA_ROADMAP_2026.md` (both in the parent `Wiping/` folder).*

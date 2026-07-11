# Reachable-arc results, manuscript updates, and pending experiments

*Living status doc. Updated as the reachable-arc matrix (`run_reach_matrix.py`) completes.*

---

## A. What changed (the two-line story)

1. The forearm's third column is the **bed-facing underside** and is physically unwipeable
   (full analysis: `REACHABILITY_ANALYSIS.md`). We scope each region to its **reachable arc**
   (`_reachable_only`), applied uniformly to all methods.
2. This makes **`task_success` meaningful again** (≈0 → ≈0.3, same policy) and closes the
   long-standing "6.9→10.4 structural gap." A small **`(s,PC2)` coverage reward term**
   (`_pc2_coverage_weight=1.0`, tuned by grid search) further lifts within-arc coverage.

---

## B. Validated cell (forearm_back, seed 1, 500-ep eval) — the template row

Ours-reach vs Yubik-reach on the reachable arc. **Report only the relevant, discriminative
metrics** (below); mean shown, SEM = std/√500 is small (all listed deltas beyond eval noise
for this seed). ⚠ = needs 3 seeds before "significant."

| Metric (better) | Ours | Yubik | winner |
|---|---|---|---|
| Targets cleared /N (↑) | **5.66** | 5.13 | Ours |
| **In-band force frac 2–12 N (↑)** | **0.161** | 0.066 | **Ours (≈2.4×)** |
| Sweep consistency (↑) | 0.143 | 0.125 | Ours |
| Clear s-coverage (↑) | 0.285 | 0.265 | Ours |
| Clear t-coverage (↑) | 0.286 | 0.240 | Ours |
| 2nd-half fraction (↑) | 0.254 | 0.162 | Ours |
| Drift-term rate (↓) | 0.204 | 0.152 | Yubik (minor) |

**Dropped from the table (with reason):**
- `pass_completions` — **contaminated**: it counts Ours' own end-of-pass reward gate firing;
  Yubik trips it incidentally by roaming to the ends. Not a valid cross-method metric. Replace
  with `sweep_consistency` (method-neutral) — already in the table.
- `mean/p95 contact force` — **misleading**: Yubik's lower force is *poking* (mostly <2 N),
  not gentleness. Force safety is reported via **in-band fraction only**.
- `task_success` shown separately (now meaningful, ≈0.3), not as a discriminator.

**Headline finding:** methods **tie on raw coverage** but **separate ~2.4× on time-in-safe-
force-band** and lead on systematic/sustained coverage → *force safety emerges from reward
structure, not force shaping.* (Coverage rows being close is the paper's point, not a weakness.)

**Honest limitation to state:** peak contact forces ~55 N (both methods, high variance) —
occasional spikes; motivates the (ablated) force-shaping term. And the drift row favours Yubik.

---

## C. The clean 9×metric matrix (publication grade) — IN PROGRESS

Running `run_reach_matrix.py` (detached, resumable): **OURS & YUBIK specialists × 4 regions ×
3 seeds**, each trained (5 chunks) then evaluated on the full battery on the reachable arc.
Flat baseline (`flat_reach`) and distilled students follow once teachers exist.

**Relevant metrics for the final table** (drop everything else to the appendix):
Targets cleared · In-band force frac · Sweep consistency · Clear s-coverage · Clear t-coverage
· 2nd-half fraction · Drift-term rate · Task success.

Status (auto-updates in `behavior_metrics_summary.csv`; regenerate the figure with
`python make_icra_figure.py` once rows land):

| cell | status |
|---|---|
| ours_reach / yubik_reach · forearm_back · s1 | ✅ done (section B) |
| ...remaining 22 cells | ⏳ training (driver running) |

**Graph:** `make_icra_figure.py` renders the method×metric figure from the CSV — run it after
the matrix fills. (Add `--full` only for the appendix.)

---

## D. Manuscript updates (drafted; collaborator recompiles the PDF)

1. **Method / Environment:** add a paragraph — *"Each region is scoped to its robot-reachable
   arc; the support-contact (bed-facing) band of a supine limb is occluded and requires
   physical re-positioning (out of scope for the wiping controller)."* Cite the 4-line
   reachability argument (no pronation DOF).
2. **Results:** replace the old matrix numbers with the reachable-arc matrix. Re-frame the
   headline: *ties on coverage, ~2.4× on in-band force.* Remove `pass_completions` and raw
   force means from the main table (reasons in B).
3. **`task_success`:** now reportable (≈0.3), no longer "≈0 for all methods."
4. **Limitations:** (i) full-circumference coverage needs limb re-positioning; (ii) peak-force
   spikes ~55 N; (iii) drift slightly higher for Ours.
5. **Negative result worth a sentence:** we tested reward-switching (Yubik→Ours→Yubik) and a
   learned spatio-temporal gate — **Ours Pareto-dominates Yubik in every state cell on the
   reachable arc** (advantage map), so no arbitration is needed. Pre-empts "did you hybridise?"

---

## E. Pending next experiments (priority order)

1. **Finish the 24-cell specialist matrix** (running) → the clean 3-seed table + figure.
2. **Flat baseline `flat_reach` × 3 seeds** → the "distillation vs conditioning" bar.
3. **Distilled students on the reachable arc** (`distill.py` from the reach teachers) — the
   headline single-policy row; re-run the +26%-vs-flat comparison with corrected stats
   (per-seed paired t **and** Wilcoxon over region×seed pairs).
4. **Stats:** once ≥3 seeds land, run `aggregate_4region.py` for the significance table.
5. **Force-spike limitation:** quantify the >12 N tail; consider re-enabling the force-shaping
   ablation rung to bound peaks.
6. **(Optional) poke-vs-pass video** of Ours vs Yubik on one reachable region — the most
   persuasive reviewer artifact.

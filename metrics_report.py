#!/usr/bin/env python3
"""
Behavior-metrics report for WipingEnv  (ICRA Claim 6: "behavior is quantitatively wiping").

Loads a trained checkpoint, rolls out N episodes, and aggregates the per-step
`info` fields the env already emits into the Claim-6 table:

    - targets_cleared / task_success / task_percentage   (headline outcome)
    - pass_completions        : # of end-of-pass evidence gates fired (reward_end_pass>0)
    - sweep_consistency       : frac of wipe-mode steps with rewarded directional motion
    - in_band_force_fraction  : frac of contact steps with normal_force in [lo,hi]
    - drift_rate              : frac of episodes terminated by the anti-drift rule
    - clear_spatial_order     : monotonicity of clear positions along the pass axis s
                                (1.0 = targets cleared in clean spatial order, ~0.5 = random)

Runs locally (needs pybullet). Mirrors learn.py's weights-only checkpoint load,
so it sidesteps the RLlib 1.x full-restore crash on mac.

Usage:
    conda activate learnbath
    cd /Users/pratyush/Desktop/Research/ROBO/Wiping/learn-bathing

    # point it at a checkpoint FILE (path must contain 'checkpoint' -> weights-only load)
    python metrics_report.py \
        --checkpoint trained_models/ARCHIVE_full_8term_winner_6p9_2026-06-12/checkpoint_000014/checkpoint-14 \
        --label full_8term_winner --episodes 100

    # squeeze run, once it exists:
    python metrics_report.py \
        --checkpoint trained_models/auto/squeeze_s1/ppo/learn_bathing:WipingEnv-v0/checkpoint_000014/checkpoint-14 \
        --label squeeze_s1 --episodes 100

Outputs:
    metrics_per_episode_<label>.csv      (one row per episode, all raw metrics)
    behavior_metrics_summary.csv         (one mean+/-std row per label; appended)
    + a markdown table printed to stdout (paste-ready for the paper)

IMPORTANT: eval clear_radius must match the headline eval (env default 0.022).
This script does NOT write _env_overrides.json, so it uses whatever is on disk.
Make sure _env_overrides.json reflects the canonical eval env, or pass --clear-radius
to force it (writes the override, restores it on exit).
"""
import argparse, csv, json, os, sys, atexit
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = "learn_bathing:WipingEnv-v0"
OVERRIDE_FILE = os.path.join(HERE, "_env_overrides.json")

# Physical constants for unit-ful metrics.
DT = 0.1                 # seconds per env step (frame_skip 5 * time_step 0.02)
FOREARM_LENGTH = 0.164   # m (0.108 + 0.028*2); s in [0,1] spans this along the arm axis


def _maybe_force_clear_radius(value, region=None, reachable_only=False):
    """Optionally pin clear_radius (and region / reachable-arc scoping) via the override file,
    restoring it on exit so we never leave a surprise env config behind for the next run."""
    if value is None and region is None and not reachable_only:
        return
    prev = None
    if os.path.exists(OVERRIDE_FILE):
        with open(OVERRIDE_FILE) as f:
            prev = f.read()
    ov = json.loads(prev) if prev else {}
    if value is not None:
        ov["clear_radius"] = value
    if region is not None:
        ov["_target_trial_order_override"] = [region]
    if reachable_only:
        ov["_reachable_only"] = True
    with open(OVERRIDE_FILE, "w") as f:
        json.dump(ov, f)
    print(f"[metrics_report] forced overrides {('clear_radius='+str(value)) if value else ''} "
          f"{('region='+region) if region else ''} in {OVERRIDE_FILE}")

    def _restore():
        if prev is None:
            try:
                os.remove(OVERRIDE_FILE)
            except OSError:
                pass
        else:
            with open(OVERRIDE_FILE, "w") as f:
                f.write(prev)
        print("[metrics_report] restored _env_overrides.json")
    atexit.register(_restore)


def episode_metrics(steps, actions, force_lo, force_hi):
    """Reduce per-step info dicts (+ actions) into one episode's metrics."""
    n = len(steps)
    wipe_steps = sum(s["wipe_mode"] for s in steps)
    contact_steps = sum(s["accepted_contact_exists"] for s in steps)

    sweep_pos = sum(1 for s in steps if s["wipe_mode"] and s["reward_sweep"] > 0)
    pass_completions = sum(1 for s in steps if s["reward_end_pass"] > 0)

    inband = sum(1 for s in steps
                 if s["accepted_contact_exists"] and force_lo <= s["normal_force"] <= force_hi)

    contact_forces = [s["normal_force"] for s in steps if s["accepted_contact_exists"]]

    # spatial order: positions s where a clear happened, in time order
    clear_s = [s["s"] for s in steps if s["clears_this_step"] > 0 and s["s"] >= 0.0]
    spatial_order = _monotonicity(clear_s)

    # --- coverage: span of arm covered while in contact (s is 0..1 along arm axis) ---
    contact_s = [s["s"] for s in steps if s["accepted_contact_exists"] and s["s"] >= 0.0]
    cov_span = (max(contact_s) - min(contact_s)) if len(contact_s) >= 2 else 0.0

    # --- effort & smoothness from the 6-D joint-delta actions ---
    A = np.asarray(actions, dtype=float) if len(actions) else np.zeros((1, 6))
    action_effort = float(np.mean(np.abs(A)))                       # mean |a| per dim
    action_smoothness = float(np.mean(np.abs(np.diff(A, axis=0)))) if len(A) > 1 else 0.0

    # --- spread / efficiency of clearing (folded in from trace_aggregate) ---
    S = np.array([s["s"] for s in steps], float)
    K = np.array([s["clears_this_step"] for s in steps], float)
    Acc = np.array([s["accepted_contact_exists"] for s in steps], int)
    ci = np.where(K > 0)[0]; ctot = float(K.sum()); B = 10
    cs_all = S[(K > 0) & (S >= 0)]
    clear_s_coverage = len(set(np.clip(cs_all * B, 0, B - 1).astype(int).tolist())) / B if len(cs_all) else float("nan")
    clear_t_coverage = len(set(np.clip(ci / n * B, 0, B - 1).astype(int).tolist())) / B if len(ci) else float("nan")
    second_half_frac = (K[n // 2:].sum() / ctot) if ctot > 0 else float("nan")
    W = 4; cset = set(ci.tolist()); contact_idx = np.where(Acc == 1)[0]
    prod = sum(1 for t in contact_idx if any((t - w) in cset or (t + w) in cset for w in range(W + 1)))
    productive_frac = (prod / len(contact_idx)) if len(contact_idx) else float("nan")
    ds = np.abs(np.diff(S, prepend=S[0])); mask = (Acc == 1) & (S >= 0)
    stall_frac = float(np.mean(ds[mask] < 0.01)) if mask.sum() else float("nan")

    # --- POKE vs WIPE: contact continuity + tangential stroke length ---
    # A "stroke" = a maximal run of steps that are IN CONTACT and SLIDING along the arm in a
    # consistent direction. Smooth wiping -> long strokes; poking (tap-in-place / jump between
    # targets) -> short strokes. n_contact_bouts counts separate contact episodes (poking
    # fragments contact); slide_frac = fraction of contact steps actually sliding (not tapping).
    dss = np.diff(S, prepend=S[0]); sgn = np.sign(dss)
    sliding = (Acc == 1) & (np.abs(dss) > 0.008)
    strokes = []; cur = 0; cdir = 0.0
    for t in range(n):
        if sliding[t] and (cdir == 0.0 or sgn[t] == cdir):
            cur += 1; cdir = sgn[t]
        else:
            if cur > 0:
                strokes.append(cur)
            cur = 1 if sliding[t] else 0
            cdir = sgn[t] if sliding[t] else 0.0
    if cur > 0:
        strokes.append(cur)
    mean_stroke_len = float(np.mean(strokes)) if strokes else 0.0
    _bd = np.diff(np.concatenate([[0], (Acc == 1).astype(int), [0]]))
    n_contact_bouts = int((_bd == 1).sum())
    slide_frac = float(np.mean(np.abs(dss[Acc == 1]) > 0.008)) if (Acc == 1).sum() else 0.0

    last = steps[-1]
    return {
        "episode_len": n,
        "episode_time_s": round(n * DT, 2),
        "targets_cleared": last["number_of_targets_cleared"],
        "task_success": last["task_success"],
        "task_percentage": round(last["task_percentage"], 2),
        # --- coverage ---
        "coverage_span_norm": round(cov_span, 4),                   # 0..1 of arm axis
        "coverage_length_m": round(cov_span * FOREARM_LENGTH, 4),   # meters along arm
        # --- timing ---
        "contact_time_s": round(contact_steps * DT, 2),
        "contact_time_fraction": round(contact_steps / max(1, n), 4),
        "wipe_time_s": round(wipe_steps * DT, 2),
        "wipe_step_fraction": round(wipe_steps / max(1, n), 4),
        # --- force (safety-relevant for assistive contact) ---
        "mean_contact_force": round(float(np.mean(contact_forces)), 4) if contact_forces else 0.0,
        "median_contact_force": round(float(np.median(contact_forces)), 4) if contact_forces else 0.0,
        "p95_contact_force": round(float(np.percentile(contact_forces, 95)), 4) if contact_forces else 0.0,
        "peak_contact_force": round(float(np.max(contact_forces)), 4) if contact_forces else 0.0,
        "force_std": round(float(np.std(contact_forces)), 4) if contact_forces else 0.0,
        "in_band_force_fraction": round(inband / max(1, contact_steps), 4),
        # --- wiping quality ---
        "pass_completions": pass_completions,
        "sweep_consistency": round(sweep_pos / max(1, wipe_steps), 4),
        "clear_spatial_order": round(spatial_order, 4),
        # --- spread / efficiency of clearing ---
        "clear_s_coverage": round(clear_s_coverage, 4),
        "clear_t_coverage": round(clear_t_coverage, 4),
        "second_half_frac": round(second_half_frac, 4),
        "productive_frac": round(productive_frac, 4),
        "stall_frac": round(stall_frac, 4),
        # --- poke vs wipe (smooth continuous stroke vs jab) ---
        "mean_stroke_len": round(mean_stroke_len, 3),
        "n_contact_bouts": n_contact_bouts,
        "slide_frac": round(slide_frac, 4),
        # --- effort / smoothness ---
        "action_effort": round(action_effort, 4),
        "action_smoothness": round(action_smoothness, 4),
        "terminated_drift": last["terminated_drift"],
        "n_clears": len(clear_s),
    }


def _monotonicity(seq):
    """Fraction of adjacent steps that move in the dominant direction.
    1.0 = perfectly ordered sweep; ~0.5 = no spatial order; nan-safe."""
    if len(seq) < 2:
        return float("nan")
    diffs = np.diff(seq)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return float("nan")
    pos = np.sum(diffs > 0)
    return max(pos, len(diffs) - pos) / len(diffs)


def summarize(rows):
    """mean +/- std across episodes for each numeric metric."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        out[k + "_mean"] = round(float(np.mean(vals)), 4) if len(vals) else float("nan")
        out[k + "_std"] = round(float(np.std(vals)), 4) if len(vals) else float("nan")
    return out


def print_markdown(label, summ):
    headline = [
        ("targets_cleared", "Targets cleared /15"),
        ("task_success", "Task success (0/1)"),
        ("coverage_length_m", "Region length covered (m)"),
        ("contact_time_s", "Time in contact (s)"),
        ("wipe_time_s", "Time wiping (s)"),
        ("mean_contact_force", "Mean contact force"),
        ("p95_contact_force", "95th-pct contact force"),
        ("peak_contact_force", "Peak contact force"),
        ("in_band_force_fraction", "In-band force fraction"),
        ("pass_completions", "Pass completions"),
        ("sweep_consistency", "Sweep consistency"),
        ("clear_spatial_order", "Clear spatial order"),
        ("action_effort", "Action effort (mean |a|)"),
        ("action_smoothness", "Action smoothness (mean |Δa|)"),
        ("terminated_drift", "Drift-termination rate"),
    ]
    print(f"\n### Behavior metrics — {label}\n")
    print("| Metric | Mean ± Std |")
    print("|---|---|")
    for key, nice in headline:
        m, s = summ.get(key + "_mean", float("nan")), summ.get(key + "_std", float("nan"))
        print(f"| {nice} | {m:.3f} ± {s:.3f} |")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to a checkpoint FILE (must contain 'checkpoint')")
    ap.add_argument("--label", required=True, help="name for this run in the outputs")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force-lo", type=float, default=0.5,
                    help="lower bound of the 'in-band' contact-force window")
    ap.add_argument("--force-hi", type=float, default=10.0,
                    help="upper bound of the 'in-band' contact-force window")
    ap.add_argument("--clear-radius", type=float, default=None,
                    help="optionally pin clear_radius for this eval (default: leave "
                         "_env_overrides.json untouched). Restored on exit.")
    ap.add_argument("--region", default=None,
                    help="optionally pin the wipe region (e.g. forearm_back, "
                         "forearm_front) so teachers/flat baseline can be scored "
                         "per-region. Restored on exit.")
    ap.add_argument("--reachable-only", action="store_true",
                    help="score on the reachable-arc scoping (drops the bed-occluded band); "
                         "MUST match how the policy was trained. Restored on exit.")
    args = ap.parse_args()

    if "checkpoint" not in args.checkpoint:
        sys.exit("ERROR: --checkpoint must be a checkpoint FILE path containing "
                 "'checkpoint' so the weights-only loader is used.")

    _maybe_force_clear_radius(args.clear_radius, args.region, args.reachable_only)

    # Import after override is set so the env picks it up at __init__.
    import multiprocessing, ray
    from learn import make_env, load_policy

    ray.init(num_cpus=multiprocessing.cpu_count(),
             ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    env = make_env(ENV_NAME, seed=args.seed)
    # render=True forces the weights-only restore path (same trick evaluate_policy uses)
    agent, _ = load_policy(env, args.algo, ENV_NAME, args.checkpoint,
                           seed=args.seed,
                           extra_configs={"num_workers": 0, "num_gpus": 0},
                           render=True)

    per_ep = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        steps, acts = [], []
        while not done:
            action = agent.compute_action(obs, explore=False)  # deterministic mean,
            # matches the student's deterministic action so effort/smoothness/force compare fairly
            obs, reward, done, info = env.step(action)
            steps.append(info); acts.append(np.asarray(action, dtype=float))
        m = episode_metrics(steps, acts, args.force_lo, args.force_hi)
        per_ep.append(m)
        if (ep + 1) % 10 == 0:
            print(f"[metrics_report] {ep+1}/{args.episodes} eps "
                  f"(last cleared={m['targets_cleared']}, "
                  f"passes={m['pass_completions']})")
        sys.stdout.flush()
    env.close()

    # per-episode CSV
    per_path = os.path.join(HERE, f"metrics_per_episode_{args.label}.csv")
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys()))
        w.writeheader()
        w.writerows(per_ep)

    # summary row (appended across labels)
    summ = summarize(per_ep)
    summ_row = {"label": args.label, "episodes": args.episodes,
                "force_band": f"[{args.force_lo},{args.force_hi}]",
                "checkpoint": args.checkpoint, **summ}
    summ_path = os.path.join(HERE, "behavior_metrics_summary.csv")
    new = not os.path.exists(summ_path)
    # robust to schema growth: rotate old summary if columns changed
    if not new:
        with open(summ_path, newline="") as f:
            existing = next(csv.reader(f), [])
        if existing != list(summ_row.keys()):
            os.rename(summ_path, summ_path + ".bak")
            new = True
    with open(summ_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ_row.keys()))
        if new:
            w.writeheader()
        w.writerow(summ_row)

    print_markdown(args.label, summ)
    print(f"[metrics_report] wrote {per_path}")
    print(f"[metrics_report] appended summary -> {summ_path}")


if __name__ == "__main__":
    main()

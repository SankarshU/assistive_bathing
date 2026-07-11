#!/usr/bin/env python
"""
clear_by_column.py — companion to reachability_probe.py.

Rolls out a trained RLlib policy (a region SPECIALIST or the flat baseline) for N episodes
and records which target COLUMNS actually get cleared, so you can read reachable% (from
reachability_probe.py) next to cleared% (here) per column. That side-by-side settles the
reachability-vs-policy question:

    column reachable ~100% but cleared ~0%   ->  POLICY / REWARD  (arm can touch it, policy won't)
    column reachable low                     ->  REACHABILITY     (arm can't touch it)

It mirrors metrics_report.py's harness exactly (same override file, ray init, weights-only
load_policy, deterministic compute_action) so the rollout is identical to how the paper
scores policies — the only addition is per-target column bookkeeping.

Mechanism: the env deletes a target from `feasible_targets` the moment it is cleared, so we
snapshot the full target set (body ids + on-arm coords) at reset, and at episode end any
body id still present was never cleared. Columns are labelled by the SAME angular grouping
the probe uses (imported `label_columns`), so column indices line up.

Run (in the `learnbath` conda env, from the repo root):
  python clear_by_column.py \
      --checkpoint trained_models/ARCHIVE_full_8term_winner_6p9_2026-06-12/checkpoint_000014/checkpoint-14 \
      --region forearm_back --episodes 100 --label ours_fb

  # then merge with the probe's reachability table for the verdict:
  python reachability_probe.py --region forearm_back --episodes 5 --json reach_fb.json
  python clear_by_column.py --checkpoint <ckpt> --region forearm_back --episodes 100 \
      --label ours_fb --reach-json reach_fb.json

NOTE: this loads RLlib checkpoints (specialists / flat baseline). The distilled *student*
is a torch model with a region-context input (distill.py) and needs that loader instead —
out of scope here; the specialist is the right policy to ask "does the policy trained ON
this region reach every column?".
"""
import argparse
import atexit
import json
import os
import sys
import numpy as np

from reachability_probe import label_columns  # shared angular column grouping

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = "learn_bathing:WipingEnv-v0"
OVERRIDE_FILE = os.path.join(HERE, "_env_overrides.json")


def _set_overrides(region, clear_radius):
    """Pin region (and optional clear_radius) via _env_overrides.json, restoring on exit —
    same mechanism metrics_report.py uses, so we never leave a surprise config behind."""
    prev = open(OVERRIDE_FILE).read() if os.path.exists(OVERRIDE_FILE) else None
    ov = json.loads(prev) if prev else {}
    ov["_target_trial_order_override"] = [region]
    if clear_radius is not None:
        ov["clear_radius"] = clear_radius
    with open(OVERRIDE_FILE, "w") as f:
        json.dump(ov, f)
    print(f"[clear_by_column] forced region={region} "
          f"{('clear_radius=' + str(clear_radius)) if clear_radius else ''} in {OVERRIDE_FILE}")

    def _restore():
        if prev is None:
            try:
                os.remove(OVERRIDE_FILE)
            except OSError:
                pass
        else:
            with open(OVERRIDE_FILE, "w") as f:
                f.write(prev)
        print("[clear_by_column] restored _env_overrides.json")
    atexit.register(_restore)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint FILE path (must contain 'checkpoint' -> weights-only load)")
    ap.add_argument("--region", required=True,
                    choices=["forearm_back", "forearm_front",
                             "upperarm_back", "upperarm_front"])
    ap.add_argument("--label", default="run", help="name for the output files")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--clear-radius", type=float, default=None,
                    help="optionally pin clear_radius (default: env default 0.022)")
    ap.add_argument("--gap-rad", type=float, default=0.5,
                    help="angular gap (rad) separating columns (match the probe)")
    ap.add_argument("--reach-json", default="",
                    help="reachability_probe.py --json output to print alongside cleared%")
    ap.add_argument("--json", default="", help="optional path to dump this summary as JSON")
    args = ap.parse_args()

    if "checkpoint" not in args.checkpoint:
        sys.exit("ERROR: --checkpoint must be a checkpoint FILE path containing 'checkpoint'.")

    _set_overrides(args.region, args.clear_radius)

    # import after the override is written so the env picks it up at __init__
    import multiprocessing, ray
    from learn import make_env, load_policy

    ray.init(num_cpus=multiprocessing.cpu_count(),
             ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    env = make_env(ENV_NAME, seed=args.seed)
    b = env.unwrapped if hasattr(env, "unwrapped") else env
    agent, _ = load_policy(env, args.algo, ENV_NAME, args.checkpoint, seed=args.seed,
                           extra_configs={"num_workers": 0, "num_gpus": 0}, render=True)

    # pooled over episodes, keyed by column index (sorted by angle, same as the probe)
    col_total = {}     # col -> #targets presented
    col_cleared = {}   # col -> #targets cleared
    col_angles = {}    # col -> list of mean angles (deg)
    ep_cleared = []

    for ep in range(args.episodes):
        obs = env.reset()
        if not getattr(b, "feasible_targets", None):
            continue
        # snapshot the full target set BEFORE any clearing
        init_bodies = [int(x) for x in b.feasible_targets]
        init_local = [np.asarray(p, dtype=float) for p in b.feasible_targets_pos]
        cols, col_angle_deg = label_columns(init_local, gap_rad=args.gap_rad)
        body_col = {bd: int(c) for bd, c in zip(init_bodies, cols)}

        done = False
        while not done:
            action = agent.compute_action(obs, explore=False)  # deterministic mean
            obs, _, done, _ = env.step(action)

        remaining = set(int(x) for x in getattr(b, "feasible_targets", []))
        cleared_here = 0
        for bd in init_bodies:
            c = body_col[bd]
            col_total[c] = col_total.get(c, 0) + 1
            col_angles.setdefault(c, []).append(col_angle_deg[c])
            if bd not in remaining:
                col_cleared[c] = col_cleared.get(c, 0) + 1
                cleared_here += 1
        ep_cleared.append(cleared_here)
        if (ep + 1) % 10 == 0:
            print(f"[clear_by_column] {ep+1}/{args.episodes} eps "
                  f"(last cleared={cleared_here})")
            sys.stdout.flush()
    env.close()

    # optional reachability table to merge
    reach = {}
    if args.reach_json and os.path.exists(args.reach_json):
        rj = json.load(open(args.reach_json))
        # match probe columns to ours by nearest mean angle
        reach = {c["col"]: c for c in rj.get("columns", [])}

    print(f"\n=== CLEARED-BY-COLUMN  {args.label}  region={args.region}  "
          f"episodes={len(ep_cleared)}  mean_cleared/ep={np.mean(ep_cleared):.2f} ===")
    header = f"  {'col':>3} {'angle(deg)':>10} {'presented':>9} {'cleared':>8} {'cleared%':>9}"
    if reach:
        header += f" {'reach%':>8}   verdict"
    print(header)
    rows = []
    cols_sorted = sorted(col_total, key=lambda c: np.mean(col_angles[c]))
    for c in cols_sorted:
        ang = float(np.mean(col_angles[c]))
        tot = col_total[c]
        cl = col_cleared.get(c, 0)
        pct = 100.0 * cl / tot if tot else 0.0
        line = f"  {c:>3} {ang:>+10.1f} {tot:>9} {cl:>8} {pct:>8.0f}%"
        rpct = None
        if reach:
            # nearest-angle match into the reachability table
            best = min(reach.values(), key=lambda rc: abs(rc["angle_deg"] - ang),
                       default=None)
            if best is not None:
                rpct = best["reachable_pct"]
                verdict = ("POLICY" if (rpct >= 90 and pct < 25) else
                           "reach-limited" if rpct < 60 else
                           "ok" if pct >= 25 else "mixed")
                line += f" {rpct:>7.0f}%   {verdict}"
        rows.append({"col": c, "angle_deg": ang, "presented": tot, "cleared": cl,
                     "cleared_pct": pct, "reach_pct": rpct})
        print(line)

    if reach:
        pol = [r for r in rows if r["reach_pct"] is not None
               and r["reach_pct"] >= 90 and r["cleared_pct"] < 25]
        if pol:
            a = ", ".join(f"{r['angle_deg']:+.0f}deg" for r in pol)
            print(f"\n  => VERDICT: POLICY/REWARD. Column(s) at {a} are >=90% reachable but "
                  f"<25% cleared — the arm can touch them, the policy won't. Fix the reward "
                  f"(add a cross-column / circumferential term to the 1-D sweep).")
        elif any(r["reach_pct"] is not None and r["reach_pct"] < 60 for r in rows):
            print("\n  => VERDICT: REACHABILITY. A low-clear column is also low-reachable — "
                  "workspace/collision limit, not the policy.")
        else:
            print("\n  => VERDICT: no clean split; inspect the table / add episodes.")

    if args.json:
        json.dump({"label": args.label, "region": args.region,
                   "episodes": len(ep_cleared),
                   "mean_cleared_per_ep": float(np.mean(ep_cleared)),
                   "columns": rows}, open(args.json, "w"), indent=2)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
eval_ladder.py — score every trained ablation rung on the FULL behavior metrics.

auto_loop only logs targets_cleared. This runs metrics_report.py on each rung's
BEST checkpoint (forearm_back, deterministic, force band [2,12]) so the ladder is
compared on coverage / passes / spatial order / force / smoothness — not just count.
Skips rungs that haven't finished training yet. Each metrics_report runs in its own
subprocess (clean ray init per policy).

    conda activate learnbath; cd learn-bathing
    python eval_ladder.py                 # all rungs that have a BEST checkpoint
    python eval_ladder.py --region forearm_back --episodes 100

Results append to behavior_metrics_summary.csv (labels like 'abl2_sweep_forearm_back').
Then: python plot_ladder.py   (or just tell the agent 'done').
"""
import argparse, glob, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def best_checkpoint(config, seed=1):
    d = os.path.join(HERE, "trained_models", "auto", f"{config}_s{seed}", "BEST")
    hits = glob.glob(os.path.join(d, "checkpoint_*", "checkpoint-*"))
    hits = [h for h in hits if not h.endswith(".tune_metadata")]
    return hits[0] if hits else None


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("al", os.path.join(HERE, "auto_loop.py"))
    al = importlib.util.module_from_spec(spec); spec.loader.exec_module(al)
    rungs = [name for name, _ in al._ABL_LADDER]

    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="forearm_back")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1, help="which trained seed (_s<seed> dir) to score")
    ap.add_argument("--force-lo", type=float, default=2.0)
    ap.add_argument("--force-hi", type=float, default=12.0)
    ap.add_argument("--configs", nargs="*", default=rungs, help="default: all ladder rungs")
    args = ap.parse_args()

    for cfg in args.configs:
        cp = best_checkpoint(cfg, args.seed)
        if cp is None:
            print(f"[eval_ladder] {cfg} (seed {args.seed}): no BEST checkpoint yet — skipping")
            continue
        # seed in the label so seeds aggregate cleanly (e.g. abl1_twophase_s2_forearm_back)
        label = f"{cfg}_s{args.seed}_{args.region}"
        print(f"[eval_ladder] scoring {cfg} -> label {label}")
        rc = subprocess.call([sys.executable, "metrics_report.py",
                              "--checkpoint", cp, "--label", label,
                              "--region", args.region, "--episodes", str(args.episodes),
                              "--clear-radius", "0.022",
                              "--force-lo", str(args.force_lo), "--force-hi", str(args.force_hi)],
                             cwd=HERE)
        if rc != 0:
            print(f"[eval_ladder] {cfg}: metrics_report FAILED rc={rc}")
    print("[eval_ladder] done. Next: python plot_ladder.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Overnight train -> eval -> log loop for WipingEnv (run on your local CPU).

Runs a sequence of configs (each = env-flag overrides), trains each in
chunks, evaluates after every chunk, appends one CSV row per eval, and
early-stops a config when it stops improving. Survives Ctrl-C between
chunks; rerunning skips configs already finished.

Usage:
    conda activate learnbath
    cd /Users/pratyush/Desktop/Research/ROBO/Wiping/learn-bathing
    python auto_loop.py                      # full overnight run
    python auto_loop.py --configs antidrift  # just one config
    python auto_loop.py --chunk 175000 --max-chunks 8

Results: auto_loop_results.csv + per-config policy dirs under trained_models/auto/
"""
import argparse, csv, importlib, json, os, re, shutil, subprocess, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "auto_loop_results.csv")
ENV_NAME = "learn_bathing:WipingEnv-v0"

# ---------------------------------------------------------------------------
# Configs to compare. Each entry: name -> dict of WipingEnv attribute overrides
# applied via env hook before training/eval. Order = run order.
# ---------------------------------------------------------------------------
CONFIGS = {
    # Full current design: 13-term + fixes A/B/C/D/E (code defaults)
    "continuity_full": {},
    # Ablation: without the new continuity fixes D/E (= previous antidrift run)
    "no_DE": {"gap_penalty_weight": 0.0, "wipe_exit_penalty": 0.0},
    # Ablation: 8-term documented reward + all fixes A-E
    "full_8term": {"use_extended_reward": False},
    # Fix F experiment: OUR frozen reward + anti-stall (penalty + termination).
    # Goal: stop the s=1.0 end-of-arm park so the policy sweeps back for remaining
    # targets — the fair test of "can our reward beat contact-counting on one arm".
    "ours_stallfix": {"use_extended_reward": False,
                      "stall_penalty_weight": 1.0, "stall_terminate": True,
                      "stall_patience": 20},
    # Original RL baseline: the 3-term contact-counting reward (git 96fb9af),
    # same env + clear_radius as the winner so the ONLY difference is the reward
    # (isolates how much the structured reward buys). For the fully-faithful
    # original constants instead, add clear_radius=0.025, task_success_threshold=0.6.
    "baseline_rl": {"use_baseline_reward": True},
    # Control: no fixes at all (the original drift-away behavior)
    "no_fixes": {
        "no_contact_penalty_weight": 0.0,
        "sticky_wipe_distance": False,
        "drift_terminate": False,
        "gap_penalty_weight": 0.0,
        "wipe_exit_penalty": 0.0,
        "wipe_start_init": False,
    },
    # Gap-closing run: SAME frozen reward as the winner (full_8term), but push
    # harder to close the 6.9 -> 10.4 success gap. Env overrides here are just
    # the winner reward; the "squeeze" levers live in HPARAMS below.
    "squeeze": {"use_extended_reward": False},

    # --- Region teachers for distillation (frozen reward; only region differs) ---
    # forearm_back teacher already exists = the archived 6.9 winner; no need to retrain.
    "region_forearm_front": {"use_extended_reward": False,
                             "_target_trial_order_override": ["forearm_front"]},
    # NOTE: upperarm_back is DISABLED — env generates its targets only at axis 1,
    # but target_axis_trial_order is hardcoded [0] => infinite reset hang. Re-enable
    # only after the env axis path is fixed. 2-region distillation for now.
    # "region_upperarm_back": {"use_extended_reward": False,
    #                          "_target_trial_order_override": ["upperarm_back"]},
    # Flat all-region baseline (the comparison reviewers ask for vs the student):
    # one policy trained on both valid regions mixed.
    "region_all": {"use_extended_reward": False,
                   "_target_trial_order_override": ["forearm_back", "forearm_front"]},
}

# Per-config hyperparameters / training levers that are NOT env-reward flags.
# Anything not listed falls back to the CLI defaults, so the original 4 ablation
# configs behave exactly as before.
#   entropy_coeff            -> PPO exploration bonus (passed to learn.py)
#   max_chunks               -> overrides --max-chunks for this config only
#   clear_radius_curriculum  -> {start, end, over_steps}: clear_radius is loosened
#                               to `start` early in training and linearly tightened
#                               to `end` by `over_steps`, then held. TRAIN ONLY.
#   eval_clear_radius        -> clear_radius forced during EVAL so every eval is
#                               comparable to the 6.9 baseline (env default 0.022).
HPARAMS = {
    "squeeze": {
        "entropy_coeff": 0.01,
        "max_chunks": 12,
        "clear_radius_curriculum": {"start": 0.030, "end": 0.022,
                                    "over_steps": 1_000_000},
        "eval_clear_radius": 0.022,
    },
}

# ---------------------------------------------------------------------------
# Incremental reward ablation (forward selection): start from the yubik baseline
# reward (distance 1.0 + action 0.01 + clears 5.0 — our env's defaults with every
# optional term OFF) and add ONE of our terms at a time. Each rung lists the terms
# still OFF; dropping a key re-enables that term at its env default. Top rung =
# our full reward. Run the ladder, plot a metric vs #terms, see where it flattens
# => which terms actually earn their place.
_ABL_OFF = {
    "use_extended_reward": False,        # extended 5 terms stay OFF throughout (shown to hurt)
    "sticky_wipe_distance": False,       # Fix B / two-phase distance
    "sweep_weight": 0.0,
    "window_weight": 0.0,
    "end_pass_weight": 0.0,
    "force_weight": 0.0, "near_target_weight": 0.0,
    "no_contact_penalty_weight": 0.0,    # Fix A
    "gap_penalty_weight": 0.0,           # Fix D
    "wipe_exit_penalty": 0.0,            # Fix E
    "drift_terminate": False,            # Fix C
}
# (name, key-or-keys to drop = enable at this rung), in build order:
_ABL_LADDER = [
    ("abl0_base_yubik",  []),
    ("abl1_twophase",    ["sticky_wipe_distance"]),
    ("abl2_sweep",       ["sweep_weight"]),
    ("abl3_window",      ["window_weight"]),
    ("abl4_endpass",     ["end_pass_weight"]),
    ("abl5_force",       ["force_weight", "near_target_weight"]),
    ("abl6_noContact",   ["no_contact_penalty_weight"]),
    ("abl7_gap",         ["gap_penalty_weight"]),
    ("abl8_wipeExit",    ["wipe_exit_penalty"]),
    ("abl9_full",        ["drift_terminate"]),   # == our full frozen reward
]


def _build_ablation_configs():
    cfgs, dropped = {}, set()
    for name, drop in _ABL_LADDER:
        dropped |= set(drop)
        cfgs[name] = {k: v for k, v in _ABL_OFF.items() if k not in dropped}
    return cfgs


CONFIGS.update(_build_ablation_configs())

# Minimal keeper reward: yubik (distance+action+clears) + ONLY the evidence-gated
# end-of-pass bonus. Everything else off. SVD arm-axis (s) + contact latch are
# inherent infra the end-pass term rides on (not separate flags). This tests the
# open question the cumulative ladder couldn't: is yubik+endpass ALONE as good as
# the up-to-endpass stack (which also had sweep+window)? = end_pass kept on.
CONFIGS["reward_minimal"] = {k: v for k, v in _ABL_OFF.items() if k != "end_pass_weight"}

# 3-way distillation teachers. forearm_back teachers already exist:
#   abl4_endpass_s1 = OURS (best reward),  baseline_rl_s1 = YUBIK (contact-counting).
# Add the matching forearm_front teachers so each reward has a 2-region teacher pair
# to distill from. (Non-RL scripted is handled separately via behavior cloning.)
CONFIGS["ours_front"]  = {**CONFIGS["abl4_endpass"], "_target_trial_order_override": ["forearm_front"]}
CONFIGS["yubik_front"] = {"use_baseline_reward": True, "_target_trial_order_override": ["forearm_front"]}

# 4-region expansion (upper-arm / shoulder). Requires the upperarm_back env-axis fix
# (2026-06-29). VALIDATE the fix with a quick render/train before trusting these.
for _reg in ["upperarm_back", "upperarm_front"]:
    CONFIGS[f"ours_{_reg}"]  = {**CONFIGS["abl4_endpass"], "_target_trial_order_override": [_reg]}
    CONFIGS[f"yubik_{_reg}"] = {"use_baseline_reward": True, "_target_trial_order_override": [_reg]}

# Flat multi-task baseline over ALL 4 regions (same OURS reward as the teachers) —
# the bar the 4-region distilled student must beat.
CONFIGS["region_all_4"] = {**CONFIGS["abl4_endpass"],
    "_target_trial_order_override": ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]}

# Reachable-arc + (s,PC2) 2-D coverage configs (2026-07). Each region is scoped to its
# robot-facing arc (drops the bed-occluded underside that no pose can wipe), and the reward
# adds the PC2 coverage term so BOTH reachable columns fill in. This is the OURS reward with
# the two new env flags; `_pc2_coverage_weight` is tunable (start 1.0, drop to 0.5 if the
# sweep distorts). Naming: ours_reach_<region> (arc only) and ours_reach_pc2_<region> (arc +
# coverage) so you can ablate the coverage term against the arc-only baseline.
for _reg in ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]:
    CONFIGS[f"ours_reach_{_reg}"] = {
        **CONFIGS["abl4_endpass"], "_target_trial_order_override": [_reg],
        "_reachable_only": True}
    CONFIGS[f"ours_reach_pc2_{_reg}"] = {
        **CONFIGS["abl4_endpass"], "_target_trial_order_override": [_reg],
        "_reachable_only": True, "_pc2_coverage_weight": 1.0, "_cov_bins": 4}
    # Yubik (contact-counting) specialist on the SAME reachable arc — the counterpart
    # policy for the Ours-vs-Yubik advantage map / switching study. The pc2 term lives in
    # the Ours reward branch only, so Yubik correctly gets no coverage shaping.
    CONFIGS[f"yubik_reach_{_reg}"] = {
        "use_baseline_reward": True, "_target_trial_order_override": [_reg],
        "_reachable_only": True}

# Flat multi-task baseline (OURS reward, no region conditioning) on the reachable arc — the
# bar the distilled student must beat, scored on the same scoped regions as the specialists.
CONFIGS["flat_reach"] = {
    **CONFIGS["abl4_endpass"], "_reachable_only": True,
    "_target_trial_order_override": ["forearm_back", "forearm_front",
                                     "upperarm_back", "upperarm_front"]}

# ---- POWERFUL specialists (OURS reward + PC2 coverage, FULL region) for the focused
# forearm_back / forearm_front study. No reachable filter (the per-reset one was unstable);
# report /15 with forearm_back's bed-facing column stated as a limitation. Trained hard
# (--max-chunks 6) via run_powerful.py; region-distilled into one strong policy.
for _reg in ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]:
    CONFIGS[f"ours_pc2_{_reg}"] = {
        **CONFIGS["abl4_endpass"], "_target_trial_order_override": [_reg],
        "_pc2_coverage_weight": 1.0, "_cov_bins": 4}

# 2-region flat baseline (same OURS+PC2 reward, NO region conditioning) — the bar the
# region-conditioned distilled student must beat in the focused forearm study.
CONFIGS["flat_forearm"] = {
    **CONFIGS["abl4_endpass"], "_pc2_coverage_weight": 1.0, "_cov_bins": 4,
    "_target_trial_order_override": ["forearm_back", "forearm_front"]}

# 4-region flat baseline (OURS+PC2 reward, all 4 arm sides, no conditioning) — the bar for
# the full 4-region distilled student.
CONFIGS["flat_pc24"] = {
    **CONFIGS["abl4_endpass"], "_pc2_coverage_weight": 1.0, "_cov_bins": 4,
    "_target_trial_order_override": ["forearm_back", "forearm_front",
                                     "upperarm_back", "upperarm_front"]}

# ---- POSE-BIN specialists (the "and poses" axis, for pose-conditioned distillation) ----
# The default supine band is randomized per episode; here we split ONE DOF (elbow, q_H index 3,
# range [0.40,1.50]) into two pose bins and train an OURS specialist per (region, pose-bin).
# These distil into pi(a | obs, region-onehot, pose-onehot). Add more bins / DOFs as needed.
POSE_BINS = {
    "poseElbBent": {3: [0.40, 0.95]},     # elbow flexed
    "poseElbExt":  {3: [0.95, 1.50]},     # elbow extended
}
for _reg in ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]:
    for _pb, _range in POSE_BINS.items():
        CONFIGS[f"ours_{_pb}_{_reg}"] = {
            **CONFIGS["abl4_endpass"], "_target_trial_order_override": [_reg],
            "_pose_range_override": _range}

OVERRIDE_FILE = os.path.join(HERE, "_env_overrides.json")


def curriculum_clear_radius(curr, steps_done):
    """Linear clear_radius schedule: start -> end over `over_steps`, then held.
    `steps_done` = timesteps already trained at the START of this chunk."""
    s, e, n = curr["start"], curr["end"], curr["over_steps"]
    if steps_done >= n:
        return e
    frac = steps_done / float(n)
    return round(s + (e - s) * frac, 4)


def write_overrides(overrides):
    with open(OVERRIDE_FILE, "w") as f:
        json.dump(overrides, f)


def run(cmd, log_path):
    """Run a command, tee output to log file, return (rc, full_output)."""
    out = []
    with open(log_path, "a") as lf:
        lf.write(f"\n===== {datetime.now().isoformat()} :: {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=HERE)
        for line in proc.stdout:
            out.append(line)
            lf.write(line)
        proc.wait()
    return proc.returncode, "".join(out)


def parse_eval(output):
    """Pull metrics from learn.py --evaluate stdout."""
    m = {}
    pats = {
        "reward_mean": r"reward mean/std:\s*([-\d.]+)",
        "task_success": r"task_success mean/std:\s*([-\d.]+)",
        "targets_cleared": r"targets_cleared mean/std:\s*([-\d.]+)",
    }
    for k, pat in pats.items():
        mm = re.search(pat, output)
        m[k] = float(mm.group(1)) if mm else float("nan")
    return m


def latest_checkpoint(policy_root, algo="ppo"):
    """learn.py saves under <save_dir>/<algo>/<env_name>/checkpoint_NNNNNN/."""
    d = os.path.join(policy_root, algo, ENV_NAME)
    if not os.path.isdir(d):
        return None
    cps = sorted(
        (c for c in os.listdir(d) if c.startswith("checkpoint_")),
        key=lambda c: int(c.split("_")[1]),
    )
    if not cps:
        return None
    last = cps[-1]
    num = int(last.split("_")[1])
    return os.path.join(d, last, f"checkpoint-{num}")


def archive_best(policy_dir, cp, score, chunk, steps):
    """Copy the current checkpoint into <policy_dir>/BEST/ so the best-scoring
    chunk survives. learn.py deletes the live checkpoint on the next save, so
    without this the peak is lost when a config declines after peaking."""
    cp_dir = os.path.dirname(cp)                 # .../checkpoint_0000NN
    best_root = os.path.join(policy_dir, "BEST")
    if os.path.isdir(best_root):
        shutil.rmtree(best_root, ignore_errors=True)
    os.makedirs(best_root, exist_ok=True)
    dst = os.path.join(best_root, os.path.basename(cp_dir))
    shutil.copytree(cp_dir, dst)
    best_cp = os.path.join(dst, os.path.basename(cp))
    with open(os.path.join(best_root, "BEST_INFO.json"), "w") as f:
        json.dump({"targets_cleared": score, "chunk": chunk, "total_steps": steps,
                   "checkpoint": best_cp,
                   "time": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    print(f"[auto_loop] archived BEST (cleared={score:.2f}, chunk {chunk}) -> {best_cp}")
    return best_cp


def append_csv(row):
    fields = list(row.keys())
    # If an existing CSV has a different (older) schema, rotate it to .bak so we
    # don't write misaligned columns. Old results are preserved, not lost.
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != fields:
            bak = CSV_PATH + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
            shutil.move(CSV_PATH, bak)
            print(f"[auto_loop] CSV schema changed; archived old results -> {bak}")
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()))
    ap.add_argument("--chunk", type=int, default=250000,
                    help="timesteps per train chunk")
    ap.add_argument("--max-chunks", type=int, default=6,
                    help="max chunks per config (chunk*max = total budget)")
    ap.add_argument("--patience", type=int, default=2,
                    help="stop config after N consecutive evals with no "
                         "improvement in targets_cleared")
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--seed", type=int, default=1,
                    help="random seed; also suffixes the per-config dir "
                         "(<name>_s<seed>) so multiple seeds coexist")
    ap.add_argument("--no-archive-best", action="store_true",
                    help="disable copying the best-scoring checkpoint to "
                         "<dir>/BEST/ (default: archive on, so the peak survives "
                         "even if the config declines after peaking)")
    ap.add_argument("--configs-file", default=None,
                    help="JSON file {name: overrides_dict, ...} merged into CONFIGS before "
                         "running (lets a hyperparameter search inject uniquely-named trials)")
    args = ap.parse_args()

    if args.configs_file:
        with open(args.configs_file) as _f:
            _extra = json.load(_f)
        CONFIGS.update(_extra)
        print(f"[auto_loop] merged {len(_extra)} config(s) from {args.configs_file}: "
              f"{list(_extra)}")

    for name in args.configs:
        overrides = CONFIGS[name]
        hp = HPARAMS.get(name, {})
        curr = hp.get("clear_radius_curriculum")
        eval_radius = hp.get("eval_clear_radius")
        entropy_coeff = hp.get("entropy_coeff")
        max_chunks = hp.get("max_chunks", args.max_chunks)
        # seed-suffixed dir keeps each seed's checkpoints + DONE marker separate
        dir_name = f"{name}_s{args.seed}"
        policy_dir = os.path.join(HERE, "trained_models", "auto", dir_name)
        os.makedirs(policy_dir, exist_ok=True)
        log_path = os.path.join(policy_dir, "run.log")
        done_marker = os.path.join(policy_dir, "DONE")
        if os.path.exists(done_marker):
            print(f"[auto_loop] {name}: already DONE, skipping")
            continue

        print(f"\n[auto_loop] ===== config '{name}' seed={args.seed} "
              f"overrides={overrides} hparams={hp} =====")

        best, stale = -1.0, 0
        best_seen = -1.0   # true max for archiving (independent of early-stop threshold)
        failed = False
        for chunk_i in range(max_chunks):
            # IMPORTANT: pass the exact checkpoint FILE for resume, not the
            # directory. A path containing "checkpoint" routes learn.py through
            # _restore_policy_weights_only(), avoiding the RLlib 1.x full-restore
            # bug (TypeError: numpy.object_ arrays in agent.restore()).
            # NOTE: learn.py deletes the previous checkpoint on each save, so
            # only the newest survives per config (by design upstream).
            resume = latest_checkpoint(policy_dir, args.algo)

            # --- TRAIN phase overrides (curriculum clear_radius if configured) ---
            train_ov = dict(overrides)
            train_radius = None
            if curr is not None:
                train_radius = curriculum_clear_radius(curr, chunk_i * args.chunk)
                train_ov["clear_radius"] = train_radius
            write_overrides(train_ov)

            train_cmd = [sys.executable, "learn.py", "--train",
                         "--algo", args.algo,
                         "--seed", str(args.seed),
                         "--train-timesteps", str(args.chunk),
                         "--save-dir", policy_dir + "/",
                         "--load-policy-path", (resume or policy_dir + "/")]
            if entropy_coeff is not None:
                train_cmd += ["--entropy-coeff", str(entropy_coeff)]
            t0 = time.time()
            rc, _ = run(train_cmd, log_path)
            if rc != 0:
                print(f"[auto_loop] {name}: TRAIN FAILED rc={rc}, see {log_path}")
                failed = True
                break

            cp = latest_checkpoint(policy_dir, args.algo)
            if cp is None:
                print(f"[auto_loop] {name}: no checkpoint found under {policy_dir} "
                      f"(check {log_path})")
                break

            # --- EVAL phase: pin clear_radius to the canonical value so every
            # eval is comparable to the 6.9 baseline (env default 0.022). ---
            eval_ov = dict(overrides)
            if eval_radius is not None:
                eval_ov["clear_radius"] = eval_radius
            write_overrides(eval_ov)

            rc, out = run([sys.executable, "learn.py", "--evaluate",
                           "--algo", args.algo, "--seed", str(args.seed),
                           "--load-policy-path", cp],
                          log_path)
            metrics = parse_eval(out)
            row = {"config": name, "seed": args.seed, "chunk": chunk_i + 1,
                   "total_steps": (chunk_i + 1) * args.chunk,
                   "train_minutes": round((time.time() - t0) / 60, 1),
                   "train_clear_radius": train_radius if train_radius is not None else "",
                   **metrics,
                   "checkpoint": cp,
                   "overrides": json.dumps(overrides),
                   "time": datetime.now().isoformat(timespec="seconds")}
            append_csv(row)
            print(f"[auto_loop] {name} chunk {chunk_i+1}: "
                  f"cleared={metrics['targets_cleared']:.2f} "
                  f"success={metrics['task_success']:.2f} "
                  f"reward={metrics['reward_mean']:.2f}")

            tc = metrics["targets_cleared"]
            # Archive the best-scoring checkpoint BEFORE learn.py overwrites it
            # next chunk. Uses a strict max so the true peak is preserved.
            if not args.no_archive_best and tc > best_seen:
                best_seen = tc
                archive_best(policy_dir, cp, tc, chunk_i + 1, (chunk_i + 1) * args.chunk)

            if tc > best + 0.25:
                best, stale = tc, 0
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"[auto_loop] {name}: plateaued "
                          f"(best cleared={best:.2f}), stopping early")
                    break

        if not failed:
            open(done_marker, "w").write(datetime.now().isoformat())
        else:
            print(f"[auto_loop] {name}: NOT marked DONE (failed); rerun will resume it")

    print(f"\n[auto_loop] all configs done. Results: {CSV_PATH}")


if __name__ == "__main__":
    main()

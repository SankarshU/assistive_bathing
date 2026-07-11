#!/usr/bin/env python3
"""
scripted_baseline.py — NON-RL wiping baseline (geometric boustrophedon sweep).

A no-learning controller: estimate the arm's long axis from the remaining target
cloud (SVD), then sweep the tool back and forth along that axis at the arm surface
via inverse kinematics. This is the classic "scripted/heuristic coverage" lower
bound that ICRA reviewers expect alongside the RL policy — it shows how much the
learned policy buys over a naive programmed wipe.

It runs THROUGH the same env/action interface as the RL policy, so it is evaluated
on identical dynamics and metrics. No reward is used (open-to-closed-loop geometry only).

  conda activate learnbath; cd learn-bathing
  echo '{"use_extended_reward": false, "clear_radius": 0.022}' > _env_overrides.json
  # quick visual check FIRST (sanity before trusting numbers):
  python scripted_baseline.py --render --episodes 1
  # then score it with the same metrics as the RL policy:
  python scripted_baseline.py --episodes 100 --metrics --label scripted_baseline

Writes (with --metrics): metrics_per_episode_<label>.csv + appends behavior_metrics_summary.csv
(reusing metrics_report's reducers), so it lands in the same comparison table.

VALIDATE before relying on it: watch one --render rollout. If the tool doesn't track
the arm, tune --press (surface offset) and --speed (sweep rate). The geometry uses
env internals (bc, robot, tool, IK) the same way the env's own reset() does.
"""
import argparse, json, os, sys, atexit
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = "learn_bathing:WipingEnv-v0"
OVERRIDE_FILE = os.path.join(HERE, "_env_overrides.json")

# action -> joint delta scaling in env.take_step: dq = frame_skip * 0.05 * action
# (frame_skip=5) => dq = 0.25 * action  => action = dq / 0.25
DQ_PER_ACTION = 0.25


def _base(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def _arm_axis(targets):
    """Principal axis (unit vec) + center of the remaining target cloud, via SVD."""
    P = np.asarray(targets, dtype=float)
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    return Vt[0], c


def _tool_pos(b):
    return np.asarray(b.bc.getLinkState(b.tool, 1, computeForwardKinematics=True,
                                        physicsClientId=b.bc._client)[0], dtype=float)


def _eef_orn(b):
    return b.bc.getLinkState(b.robot.id, b.robot.eef_id, computeForwardKinematics=True,
                             physicsClientId=b.bc._client)[1]


def scripted_action(b, phase, press, speed):
    """Compute a 6-D action that drives the tool toward a sweeping waypoint along
    the arm axis. Returns (action[6], new_phase)."""
    targets = getattr(b, "feasible_targets_pos_world", None)
    if not targets:                       # nothing left: hold still
        return np.zeros(b.action_robot_len, dtype=np.float32), phase

    axis, center = _arm_axis(targets)
    P = np.asarray(targets, dtype=float)
    s = (P - center) @ axis
    s_lo, s_hi = float(s.min()), float(s.max())

    # triangle wave in [0,1] -> position along axis; advance phase by speed
    phase = (phase + speed) % 2.0
    tri = phase if phase <= 1.0 else (2.0 - phase)
    s_target = s_lo + tri * (s_hi - s_lo)

    # waypoint: point on the axis at s_target, pulled toward the cloud (press = surface bias)
    radial = center - (center @ axis) * axis      # not exact normal; small inward bias
    rn = np.linalg.norm(radial)
    inward = (radial / rn) if rn > 1e-6 else np.zeros(3)
    waypoint = center + s_target * axis - press * inward

    # IK to the waypoint (same call the env uses in reset)
    q_ik = b.bc.calculateInverseKinematics(
        b.robot.id, b.robot.eef_id, waypoint.tolist(), _eef_orn(b),
        physicsClientId=b.bc._client, maxNumIterations=60, residualThreshold=1e-4)
    q_ik = np.asarray(q_ik[:b.action_robot_len], dtype=float)

    cur = np.asarray([x[0] for x in b.bc.getJointStates(
        b.robot.id, jointIndices=b.robot.arm_controllable_joints,
        physicsClientId=b.bc._client)], dtype=float)

    dq = q_ik - cur
    action = np.clip(dq / DQ_PER_ACTION, -1.0, 1.0).astype(np.float32)
    return action, phase


def _set_env(region, clear_radius):
    prev = open(OVERRIDE_FILE).read() if os.path.exists(OVERRIDE_FILE) else None
    ov = {"use_extended_reward": False, "clear_radius": clear_radius}
    if region:
        ov["_target_trial_order_override"] = [region]
    with open(OVERRIDE_FILE, "w") as f:
        json.dump(ov, f)
    def _restore():
        if prev is None:
            try: os.remove(OVERRIDE_FILE)
            except OSError: pass
        else:
            open(OVERRIDE_FILE, "w").write(prev)
    atexit.register(_restore)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--region", default="forearm_back")
    ap.add_argument("--clear-radius", type=float, default=0.022)
    ap.add_argument("--press", type=float, default=0.01, help="inward surface bias (m)")
    ap.add_argument("--speed", type=float, default=0.04, help="sweep phase advance/step")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--metrics", action="store_true", help="score with metrics_report reducers")
    ap.add_argument("--label", default="scripted_baseline")
    ap.add_argument("--force-lo", type=float, default=0.5)
    ap.add_argument("--force-hi", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--collect", default=None,
                    help="also save (obs, action, context) to this .npz for distillation "
                         "(behavior cloning of the scripted controller)")
    ap.add_argument("--contexts", default="",
                    help="comma-list of regions defining the one-hot context order for "
                         "--collect (must match the student's context; default = 2-region)")
    args = ap.parse_args()
    REGIONS_ORDER = (args.contexts.split(",") if args.contexts
                     else ["forearm_back", "forearm_front"])   # must match distill context order

    _set_env(args.region, args.clear_radius)
    import multiprocessing, ray
    from learn import make_env
    if args.metrics:
        import importlib.util
        spec = importlib.util.spec_from_file_location("mr", os.path.join(HERE, "metrics_report.py"))
        mr = importlib.util.module_from_spec(spec); spec.loader.exec_module(mr)

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    env = make_env(ENV_NAME, seed=args.seed)
    if args.render:
        env.render()
    b = _base(env)

    cvec = np.zeros(len(REGIONS_ORDER), np.float32)
    if args.region in REGIONS_ORDER:
        cvec[REGIONS_ORDER.index(args.region)] = 1.0
    obs_buf, act_buf = [], []

    per_ep, cleared_all = [], []
    for ep in range(args.episodes):
        obs = env.reset(); done = False; phase = 0.0
        steps, acts = [], []
        while not done:
            action, phase = scripted_action(b, phase, args.press, args.speed)
            if args.collect:
                obs_buf.append(np.asarray(obs, np.float32)); act_buf.append(np.asarray(action, np.float32))
            obs, _, done, info = env.step(action)
            steps.append(info); acts.append(action)
        cleared_all.append(info.get("number_of_targets_cleared", 0))
        if args.metrics:
            per_ep.append(mr.episode_metrics(steps, acts, args.force_lo, args.force_hi))
        if (ep + 1) % 10 == 0:
            print(f"[scripted] {ep+1}/{args.episodes} cleared={cleared_all[-1]}")
        sys.stdout.flush()
    env.close()

    print(f"[scripted] region={args.region} mean_cleared={np.mean(cleared_all):.2f} "
          f"± {np.std(cleared_all):.2f}")

    if args.collect and obs_buf:
        os.makedirs(os.path.dirname(args.collect) or ".", exist_ok=True)
        n = len(obs_buf)
        # same npz schema as distill.collect; deterministic controller -> log_std = 0 (BC only)
        np.savez_compressed(args.collect,
            obs=np.array(obs_buf, np.float32), mean=np.array(act_buf, np.float32),
            log_std=np.zeros((n, act_buf[0].shape[0]), np.float32),
            context=np.tile(cvec, (n, 1)).astype(np.float32),
            region=args.region, regions=np.array(REGIONS_ORDER))
        print(f"[scripted] collected {n} steps (region {args.region}) -> {args.collect}")

    if args.metrics and per_ep:
        import csv
        per_path = os.path.join(HERE, f"metrics_per_episode_{args.label}.csv")
        with open(per_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys())); w.writeheader(); w.writerows(per_ep)
        summ = mr.summarize(per_ep)
        mr.print_markdown(args.label, summ)
        print(f"[scripted] wrote {per_path}")
        # append to the SHARED summary table (same file/schema as metrics_report & distill eval)
        summ_row = {"label": args.label, "episodes": args.episodes,
                    "force_band": f"[{args.force_lo},{args.force_hi}]",
                    "checkpoint": "scripted_baseline", **summ}
        summ_path = os.path.join(HERE, "behavior_metrics_summary.csv")
        new = not os.path.exists(summ_path)
        if not new:
            with open(summ_path, newline="") as f:
                existing = next(csv.reader(f), [])
            if existing != list(summ_row.keys()):
                os.rename(summ_path, summ_path + ".bak"); new = True
        with open(summ_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ_row.keys()))
            if new: w.writeheader()
            w.writerow(summ_row)
        print(f"[scripted] appended summary -> {summ_path}")


if __name__ == "__main__":
    main()

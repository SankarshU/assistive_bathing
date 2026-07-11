#!/usr/bin/env python
"""
pose_feasibility.py — acceptance test for the env pose-roll hook (_apply_pose_roll).

Goal: confirm that re-posing the limb (rolling it about the arm axis) EXPOSES the column
that is bed-occluded in the nominal supine pose, collision-free, and find the minimal set of
poses that jointly cover all columns -> the 'pose' context axis for distillation.

Two phases (efficient, hang-safe):
  PHASE 1  identify the roll DOF geometrically. Perturb each controllable human arm joint and
           measure how much it tilts the forearm's LONG axis. The roll-about-arm-axis joint
           (shoulder internal/external rotation) is the one that rotates the limb while
           BARELY tilting the long axis (min delta-axis) -> that is the joint we roll.
  PHASE 2  sweep roll angles on that joint via the real env hook, run the raster through
           env.step() (the true clearing machinery), and report clearable% per column for
           each pose. Column labels come from LOCAL (on-arm) coords, which are invariant to
           the roll, so col@98 / col@155 / col@213 are consistent across poses. A pre-check
           (manual roll on the current sample) skips collision-only angles so the env's
           resample loop can't hang.

Run (in the `learnbath` conda env, from the repo root):
  python pose_feasibility.py --region forearm_back --episodes 3 --json pose_fb.json
"""
import argparse
import json
import numpy as np
import gym
import learn_bathing  # noqa: F401

from reachability_probe import label_columns
from scripted_raster import raster_action


def _forearm_axis(b):
    """Forearm capsule long axis in world (same construction the env uses to lay targets)."""
    pos, orn = b.bc.getLinkState(b.humanoid._humanoid, b.right_elbow,
                                 computeForwardKinematics=True, physicsClientId=b.bc._client)[4:6]
    orn2 = b.util.rotate_quaternion_by_axis(orn, axis="x", degrees=-90)
    R = np.array(b.bc.getMatrixFromQuaternion(orn2)).reshape(3, 3)
    a = R @ np.array([0.0, 0.0, 1.0])
    return a / (np.linalg.norm(a) + 1e-12)


def column_normals(b):
    """Mean OUTWARD surface normal (world) per local column, and a human-readable facing.
    Tells us what each column points at -> why the tool (from +x, above) can/can't reach it."""
    world = np.asarray([np.asarray(p, float) for p in b.feasible_targets_pos_world])
    local = [np.asarray(p, float) for p in b.feasible_targets_pos]
    cols, col_angle = label_columns(local)
    c = world.mean(axis=0)
    _, _, Vt = np.linalg.svd(world - c, full_matrices=False)
    axis = Vt[0]
    out = {}
    for cc in sorted(set(cols.tolist()), key=lambda k: col_angle[k]):
        P = world[cols == cc]
        radial = P - (c + ((P - c) @ axis)[:, None] * axis)   # perp to arm axis = outward
        n = radial.mean(axis=0)
        n = n / (np.linalg.norm(n) + 1e-12)
        facing = []
        facing.append("+x/robot" if n[0] > 0.4 else ("-x/torso" if n[0] < -0.4 else ""))
        facing.append("up" if n[2] > 0.4 else ("down/bed" if n[2] < -0.4 else ""))
        out[cc] = (col_angle[cc], n, " ".join(f for f in facing if f) or "lateral")
    return out


def identify_roll_joint(b, joints, probe=0.2):
    """Return the joint whose small perturbation least tilts the forearm long axis
    (= roll-about-arm-axis), with per-joint diagnostics."""
    a0 = _forearm_axis(b)
    diag = []
    for j in joints:
        cur = b.bc.getJointState(b.humanoid._humanoid, j, physicsClientId=b.bc._client)[0]
        b.bc.resetJointState(b.humanoid._humanoid, j, cur + probe, physicsClientId=b.bc._client)
        b.bc.stepSimulation(physicsClientId=b.bc._client)
        aj = _forearm_axis(b)
        d_axis = float(np.degrees(np.arccos(min(1.0, abs(float(a0 @ aj))))))
        b.bc.resetJointState(b.humanoid._humanoid, j, cur, physicsClientId=b.bc._client)
        b.bc.stepSimulation(physicsClientId=b.bc._client)
        name = b.bc.getJointInfo(b.humanoid._humanoid, j,
                                 physicsClientId=b.bc._client)[1].decode()
        diag.append((j, name, d_axis))
    diag.sort(key=lambda x: x[2])          # smallest axis-tilt first = the roll DOF
    return diag[0][0], diag


def _roll_collision_free(b, joint, delta, samples=4):
    """Pre-check: does rolling by delta stay collision-free for at least one sampled human
    config? Avoids the env resample-loop hanging on an always-colliding angle."""
    for _ in range(samples):
        # apply the roll on the CURRENT locked config and test for collision
        cur = b.bc.getJointState(b.humanoid._humanoid, joint, physicsClientId=b.bc._client)[0]
        b.bc.resetJointState(b.humanoid._humanoid, joint, cur + delta, physicsClientId=b.bc._client)
        b.bc.stepSimulation(physicsClientId=b.bc._client)
        ok = not b.human_in_collision()
        b.bc.resetJointState(b.humanoid._humanoid, joint, cur, physicsClientId=b.bc._client)
        b.bc.stepSimulation(physicsClientId=b.bc._client)
        if ok:
            return True
    return False


def run(region, episodes, press, rolls_deg, joints, verbose):
    env = gym.make("WipingEnv-v0")
    b = env.unwrapped if hasattr(env, "unwrapped") else env
    b._target_trial_order_override = [region]

    # ---- Phase 1: identify the roll DOF + read each column's facing ----
    env.reset()
    roll_joint, diag = identify_roll_joint(b, list(b.human_controllable_joints))
    print("\n=== PHASE 1: arm DOFs (delta forearm-axis tilt per joint) ===")
    for j, name, d in diag:
        tag = "  <- least tilt (roll-like)" if j == roll_joint else ""
        print(f"  joint {j:>2} {name:22s} axis-tilt={d:5.1f}deg{tag}")
    print("\n=== column facing (outward normal, nominal pose) -> what each column points at ===")
    for cc, (a, n, facing) in column_normals(b).items():
        print(f"  col@{a:+.0f}: normal=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  faces: {facing}")

    # ---- Phase 2: sweep EVERY arm DOF (a re-pose need not be a roll), measure per-column
    # clearable% via the real env.step()+raster path ----
    sweep_joints = joints if joints else list(b.human_controllable_joints)
    poses = [("nominal", None, 0.0)] + [(f"j{j}{d:+d}", j, np.radians(d))
                                        for j in sweep_joints for d in rolls_deg]
    ang, cov = {}, {}
    for label, j, delta in poses:
        # pre-check feasibility (skip angles that always collide -> would hang reset)
        if j is not None:
            b._pose_roll_joint, b._pose_roll_delta = None, 0.0
            env.reset()
            if not _roll_collision_free(b, j, delta):
                print(f"  [{label}] SKIPPED — roll collides with bed for sampled configs")
                cov[label] = None
                continue
        b._pose_roll_joint, b._pose_roll_delta = j, float(delta)
        cov[label] = {"tot": {}, "cl": {}}
        for ep in range(episodes):
            env.reset()                         # applies the pose via the env hook
            if not getattr(b, "feasible_targets", None):
                continue
            init_bodies = [int(x) for x in b.feasible_targets]
            init_local = [np.asarray(p, float) for p in b.feasible_targets_pos]
            cols, col_angle = label_columns(init_local)
            body_col = {bd: int(c) for bd, c in zip(init_bodies, cols)}
            done, info = False, {}
            while not done:
                _, _, done, info = env.step(raster_action(b, press))
            remaining = set(int(x) for x in getattr(b, "feasible_targets", []))
            for bd in init_bodies:
                cc = body_col[bd]
                ang[cc] = col_angle[cc]
                cov[label]["tot"][cc] = cov[label]["tot"].get(cc, 0) + 1
                if bd not in remaining:
                    cov[label]["cl"][cc] = cov[label]["cl"].get(cc, 0) + 1
            if verbose:
                print(f"    [{label}] ep{ep} cleared={info.get('number_of_targets_cleared', 0)}")
    b._pose_roll_joint, b._pose_roll_delta = None, 0.0
    env.close()

    def pct(label, c):
        d = cov.get(label)
        if not d:
            return float("nan")
        t = d["tot"].get(c, 0)
        return 100.0 * d["cl"].get(c, 0) / t if t else float("nan")

    all_cols = sorted(ang, key=lambda c: ang[c])
    print("\n=== PHASE 2: pose x column clearable%  (columns are pose-invariant local bands) ===")
    print("  " + "pose".ljust(10) + "".join(f"col@{ang[c]:+.0f}".rjust(12) for c in all_cols))
    table = {}
    for label, *_ in poses:
        if cov.get(label) is None:
            continue
        table[label] = {c: pct(label, c) for c in all_cols}
        cells = "".join(f"{(0.0 if np.isnan(table[label][c]) else table[label][c]):>10.0f}%" for c in all_cols)
        print("  " + label.ljust(10) + cells)

    # minimal covering pose set (greedy, >=30%)
    THRESH = 30.0
    remaining, chosen, labels = set(all_cols), [], list(table)
    while remaining:
        best, gain = None, -1
        for lbl in labels:
            g = sum(1 for c in remaining if table[lbl].get(c, 0) >= THRESH)
            if g > gain:
                best, gain = lbl, g
        if gain <= 0:
            break
        chosen.append(best)
        remaining -= {c for c in remaining if table[best].get(c, 0) >= THRESH}
        labels.remove(best)

    print("\n  VERDICT:")
    if not remaining:
        print(f"    * ALL columns covered by poses {chosen} (roll joint {roll_joint}).")
        print(f"      -> pose context axis confirmed. Train specialists per (region,pose) in")
        print(f"         {chosen}, distill into pi(a|obs,region,pose), orchestrate re-pose at deploy.")
    else:
        uncov = [f"col@{ang[c]:+.0f}" for c in remaining]
        print(f"    * poses {chosen} still leave {uncov} uncovered — widen roll range "
              f"(--rolls) or check the roll joint / q_H range.")

    return {"region": region, "roll_joint": int(roll_joint),
            "phase1": [{"joint": j, "name": n, "axis_tilt_deg": d} for j, n, d in diag],
            "columns": {str(c): {"angle_deg": ang[c],
                                 "clearable_pct": {l: table[l].get(c) for l in table}}
                        for c in all_cols},
            "covering_poses": chosen, "uncovered": sorted(remaining)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="forearm_back",
                    choices=["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--press", type=float, default=0.01)
    ap.add_argument("--rolls", type=int, nargs="+", default=[60, 90, -60, -90],
                    help="angles (deg) to test on each swept joint")
    ap.add_argument("--joints", type=int, nargs="+", default=None,
                    help="human arm joints to sweep (default: all controllable [3,4,5,7])")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = run(args.region, args.episodes, args.press, args.rolls, args.joints, args.verbose)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2, default=float)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()

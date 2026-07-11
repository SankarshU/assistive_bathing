#!/usr/bin/env python
"""
reachability_probe.py — POLICY-INDEPENDENT per-target / per-column reachability test.

Question this answers: when the trained policy wipes only ~2 of the ~3 angular
columns of a region and leaves one uncovered, is that missed column
  (a) physically UNREACHABLE by the UR5 + tool (workspace / collision / joint-limit), or
  (b) reachable, and the policy simply never goes there (reward / training issue)?

Method (no learning involved):
  1. Reset the real WipingEnv for one region -> get its ~15 targets (world + on-arm coords).
  2. Group the targets into angular COLUMNS around the arm (the same "columns" you see).
  3. For EACH target, ask geometry only: is there a robot config that places the TOOL TIP
     within clear_radius of it, in joint-limits and collision-free? (IK + the env's own
     robot_in_collision check; several tool orientations are tried, target counts as
     reachable if ANY succeeds.)
  4. Aggregate over N episodes (the human arm pose is re-randomized each reset) and print a
     per-column reachable-% table.

Reading the result:
  * One column at a much lower reachable-% than the others  -> REACHABILITY-limited.
      The policy can't be blamed for a target the arm can't touch. Fix by widening the
      region split, relaxing approach orientation, or excluding those targets from "/15".
  * All columns ~100% reachable                             -> POLICY / REWARD.
      The missed column is reachable; the 1-D sweep reward (projection onto the arm's long
      axis only) gives no incentive to move across columns. Fix on the reward side.

Run (in the `learnbath` conda env, from the repo root):
  python reachability_probe.py --region forearm_back --episodes 5
  python reachability_probe.py --region upperarm_back --episodes 5 --verbose
  # all four:
  for r in forearm_back forearm_front upperarm_back upperarm_front; do
      python reachability_probe.py --region $r --episodes 5; done

NOTE: robot_in_collision (the env's own checker) tests the ARM vs bed/cube/human + self;
the soft tool is not in that set (identical to how the env decides feasibility). Reported
tool_gap is the IK residual between the commanded tool-tip pose and the target.
"""
import argparse
import json
import numpy as np
import gym
import learn_bathing  # noqa: F401  (registers WipingEnv-v0)


# ----------------------------------------------------------------------------- helpers
def _base(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def label_columns(local_pts, gap_rad=0.5):
    """Group targets into angular columns around the arm's long axis.

    Targets are generated in the arm-link frame with the capsule long axis along local z
    (see WipingEnv.generate_targets / util.capsule_points), so local (x, y) is the
    cross-section plane and atan2(y, x) is the angle around the arm."""
    P = np.asarray(local_pts, dtype=float)
    ang = np.arctan2(P[:, 1], P[:, 0])
    # rotate so the cluster is centred at 0 (avoids the +/-pi wrap splitting a column)
    cmean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    a = (ang - cmean + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(a)
    cols = np.empty(len(a), dtype=int)
    c = 0
    cols[order[0]] = 0
    for k in range(1, len(order)):
        if a[order[k]] - a[order[k - 1]] > gap_rad:
            c += 1
        cols[order[k]] = c
    # absolute mean angle (deg) per column, for a stable left->right label across episodes
    col_angle_deg = {}
    for cc in range(c + 1):
        m = cols == cc
        col_angle_deg[cc] = float(np.degrees(
            np.arctan2(np.sin(ang[m]).mean(), np.cos(ang[m]).mean())))
    return cols, col_angle_deg


def tool_tip_offset(b):
    """Rigid transform eef_frame -> tool-tip (link 1), calibrated from the live pose."""
    eef = b.bc.getLinkState(b.robot.id, b.robot.eef_id,
                            computeForwardKinematics=True, physicsClientId=b.bc._client)[:2]
    tip = b.bc.getLinkState(b.tool, 1,
                            computeForwardKinematics=True, physicsClientId=b.bc._client)[:2]
    inv_eef = b.bc.invertTransform(eef[0], eef[1])
    eef_to_tip = b.bc.multiplyTransforms(inv_eef[0], inv_eef[1], tip[0], tip[1])
    tip_to_eef = b.bc.invertTransform(eef_to_tip[0], eef_to_tip[1])
    return eef_to_tip, tip_to_eef


def _orn_grid(b, base_orn, tilt_max, tilt_step):
    """DENSE grid of tool orientations: the region's approach orientation rotated by every
    (tx about x, ty about y) pair in [-tilt_max, tilt_max]. A target is called reachable
    only if NO orientation in this whole grid can place the tool on it — so a low
    reachable% is a real geometric limit, not an under-searched approach angle."""
    vals = list(range(-tilt_max, tilt_max + 1, tilt_step))
    cands = []
    for tx in vals:
        qx = base_orn if tx == 0 else b.util.rotate_quaternion_by_axis(base_orn, axis="x", degrees=tx)
        for ty in vals:
            cands.append(qx if ty == 0 else b.util.rotate_quaternion_by_axis(qx, axis="y", degrees=ty))
    return cands


def _collision_bodies(b, q):
    """Name what the arm collides with at config q (bed / human / cube / self), so an
    unreachable underside target can be attributed to a real obstruction."""
    for i, j in enumerate(b.robot.arm_controllable_joints):
        b.bc.resetJointState(b.robot.id, j, q[i], physicsClientId=b.bc._client)
    b.bc.stepSimulation(physicsClientId=b.bc._client)
    hit = []
    for name, body in (("bed", getattr(b, "bed_id", None)),
                       ("human", getattr(b.humanoid, "_humanoid", None)),
                       ("cube", getattr(b, "cube_id", None))):
        if body is None:
            continue
        pts = b.bc.getClosestPoints(b.robot.id, body, distance=0.0,
                                    physicsClientId=b.bc._client)
        if pts:
            hit.append(name)
    return hit or ["self/other"]


def reach_one_target(b, target_world, eef_to_tip, tip_to_eef, base_orn, clear_radius, seed_q,
                     tilt_max, tilt_step):
    """Best-case reachability of a single target. Returns dict with reachable flag +
    diagnostics. Tries several tool orientations; success if any is in-limits,
    collision-free, and lands the tool tip within clear_radius."""
    lo = b.robot.arm_lower_limits
    hi = b.robot.arm_upper_limits
    n = len(b.robot.arm_controllable_joints)
    best = {"reachable": False, "reason": "joint-limit", "blocker": None,
            "tool_gap": np.inf, "ik_err": np.inf}
    # failure evidence collected across the whole orientation grid
    saw_collision_q = None      # an in-limits pose that reaches the target but collides
    saw_workspace = False       # an in-limits pose exists but IK can't place the eef there

    for orn in _orn_grid(b, base_orn, tilt_max, tilt_step):
        # reseed IK from a known-good region config for a consistent starting guess
        for i, j in enumerate(b.robot.arm_controllable_joints):
            b.bc.resetJointState(b.robot.id, j, seed_q[i], physicsClientId=b.bc._client)

        # eef pose that places the tool tip AT the target with orientation `orn`
        world_to_eef = b.bc.multiplyTransforms(target_world, orn, tip_to_eef[0], tip_to_eef[1])
        q = b.bc.calculateInverseKinematics(
            b.robot.id, b.robot.eef_id, world_to_eef[0], world_to_eef[1],
            lowerLimits=lo, upperLimits=hi,
            jointRanges=b.robot.arm_joint_ranges, restPoses=b.robot.arm_rest_poses,
            maxNumIterations=60, residualThreshold=1e-4, physicsClientId=b.bc._client)
        q = [q[i] for i in range(n)]
        in_limits = (min(q) >= min(lo)) and (max(q) <= max(hi))

        for i, j in enumerate(b.robot.arm_controllable_joints):
            b.bc.resetJointState(b.robot.id, j, q[i], physicsClientId=b.bc._client)
        b.bc.stepSimulation(physicsClientId=b.bc._client)

        eef_pos = b.bc.getLinkState(b.robot.id, b.robot.eef_id,
                                    computeForwardKinematics=True, physicsClientId=b.bc._client)[0]
        ik_err = float(np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos)))
        # predicted tool tip from the rigid eef->tip offset (exact; avoids constraint settle)
        eef_full = b.bc.getLinkState(b.robot.id, b.robot.eef_id,
                                     computeForwardKinematics=True, physicsClientId=b.bc._client)[:2]
        tip_pred = b.bc.multiplyTransforms(eef_full[0], eef_full[1],
                                           eef_to_tip[0], eef_to_tip[1])[0]
        tool_gap = float(np.linalg.norm(np.array(tip_pred) - np.array(target_world)))
        placed = in_limits and (ik_err <= 0.03) and (tool_gap <= clear_radius)
        collided = bool(b.robot_in_collision(q)) if placed else False

        if placed and not collided:
            return {"reachable": True, "reason": "ok", "blocker": None,
                    "tool_gap": tool_gap, "ik_err": ik_err}
        # record why this orientation failed (priority: collision > workspace > limit)
        if placed and collided and saw_collision_q is None:
            saw_collision_q = list(q)
        elif in_limits and not placed:
            saw_workspace = True
        if tool_gap < best["tool_gap"]:
            best.update(tool_gap=tool_gap, ik_err=ik_err)

    # no orientation worked -> classify the closest-to-working failure
    if saw_collision_q is not None:
        best["reason"] = "collision"        # kinematically reachable, but the arm hits something
        best["blocker"] = ",".join(_collision_bodies(b, saw_collision_q))
    elif saw_workspace:
        best["reason"] = "workspace"        # in joint limits, but IK can't place the eef there
    else:
        best["reason"] = "joint-limit"      # no in-limit IK solution at all
    return best


def run_region(region, episodes, clear_radius, gap_rad, verbose, tilt_max, tilt_step):
    env = gym.make("WipingEnv-v0")
    b = _base(env)
    b._target_trial_order_override = [region]

    per_col = {}       # col_id -> list of reachable bools (pooled over episodes)
    per_col_ang = {}   # col_id -> list of mean angles (deg)
    per_col_gap = {}   # col_id -> list of tool_gaps
    per_col_reason = {}  # col_id -> list of failure reasons (unreachable only)
    per_col_block = {}   # col_id -> list of blocker names (collision failures only)
    n_targets_seen = []

    for ep in range(episodes):
        env.reset()
        eef_to_tip, tip_to_eef = tool_tip_offset(b)

        world = [np.asarray(p, dtype=float) for p in b.feasible_targets_pos_world]
        local = [np.asarray(p, dtype=float) for p in b.feasible_targets_pos]
        n_targets_seen.append(len(world))
        cols, col_angle_deg = label_columns(local, gap_rad=gap_rad)

        base_orn = b.world_to_target_point[1]          # region's canonical approach orientation
        seed_q = getattr(b, "init_q_R", None) or [0.0] * len(b.robot.arm_controllable_joints)

        if verbose:
            print(f"\n  [ep {ep}] region={region} n_targets={len(world)} "
                  f"columns={len(col_angle_deg)}")
        for idx, (tw, cid) in enumerate(zip(world, cols)):
            r = reach_one_target(b, tw, eef_to_tip, tip_to_eef, base_orn, clear_radius, seed_q,
                                 tilt_max, tilt_step)
            per_col.setdefault(cid, []).append(r["reachable"])
            per_col_ang.setdefault(cid, []).append(col_angle_deg[cid])
            per_col_gap.setdefault(cid, []).append(min(r["tool_gap"], 1.0))
            if not r["reachable"]:
                per_col_reason.setdefault(cid, []).append(r["reason"])
                if r["blocker"]:
                    per_col_block.setdefault(cid, []).append(r["blocker"])
            if verbose:
                tag = "OK  " if r["reachable"] else r["reason"][:4].upper()
                blk = f" blk={r['blocker']}" if r.get("blocker") else ""
                print(f"      t{idx:02d} col{cid} ang={col_angle_deg[cid]:+6.1f}deg "
                      f"{tag}  gap={r['tool_gap']:.4f} ik_err={r['ik_err']:.4f}{blk}")
    env.close()

    # ---- report ---------------------------------------------------------------
    print(f"\n=== REGION {region} | clear_radius={clear_radius} | "
          f"episodes={episodes} | avg_targets={np.mean(n_targets_seen):.1f} ===")
    def _mode(xs):
        return max(set(xs), key=xs.count) if xs else ""

    print(f"  {'col':>3} {'angle(deg)':>10} {'n':>4} {'reachable%':>11} "
          f"{'mean_gap(m)':>12}  {'why-unreachable':<16} blocker")
    rows = []
    for cid in sorted(per_col, key=lambda c: np.mean(per_col_ang[c])):
        arr = np.array(per_col[cid], dtype=float)
        ang = float(np.mean(per_col_ang[cid]))
        pct = 100.0 * arr.mean()
        gap = float(np.mean(per_col_gap[cid]))
        reason = _mode(per_col_reason.get(cid, []))
        blocker = _mode(per_col_block.get(cid, []))
        rows.append((cid, ang, len(arr), pct, gap, reason, blocker))
        print(f"  {cid:>3} {ang:>+10.1f} {len(arr):>4} {pct:>10.0f}% {gap:>12.4f}  "
              f"{reason:<16} {blocker}")

    pcts = [r[3] for r in rows]
    if len(pcts) >= 2:
        spread = max(pcts) - min(pcts)
        worst = min(rows, key=lambda r: r[3])
        print("\n  VERDICT HINT:")
        if min(pcts) < 60 and spread > 30:
            why = worst[5] + (f" ({worst[6]})" if worst[6] else "")
            print(f"    -> REACHABILITY-limited. Column at {worst[1]:+.0f}deg is only "
                  f"{worst[3]:.0f}% reachable vs {max(pcts):.0f}% for the best column;")
            print(f"       dominant failure = {why}. A dense orientation grid "
                  f"(+/-{tilt_max}deg) was searched, so this is a real geometric limit.")
            print(f"       Fix on the ENV/region side (exclude these targets, relax the "
                  f"region split), not the policy.")
        elif min(pcts) >= 90:
            print(f"    -> POLICY / REWARD. All columns >=90% reachable "
                  f"(min {min(pcts):.0f}%). The missed column IS reachable; the miss is a")
            print(f"       training/reward issue (1-D sweep axis gives no cross-column "
                  f"incentive). Confirm with the per-target clear logger.")
        else:
            print(f"    -> MIXED. min column {min(pcts):.0f}%, spread {spread:.0f}%. Likely "
                  f"BOTH a reach-limited column and a reachable-but-skipped one — read the "
                  f"table + cross-check clear_by_column.py.")
    return {"region": region, "clear_radius": clear_radius, "episodes": episodes,
            "tilt_max": tilt_max, "tilt_step": tilt_step,
            "columns": [{"col": int(r[0]), "angle_deg": float(r[1]), "n": int(r[2]),
                         "reachable_pct": float(r[3]), "mean_gap_m": float(r[4]),
                         "reason": r[5], "blocker": r[6]}
                        for r in rows]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="forearm_back",
                    choices=["forearm_back", "forearm_front",
                             "upperarm_back", "upperarm_front"])
    ap.add_argument("--episodes", type=int, default=5,
                    help="resets to average over (arm pose is re-randomized each reset)")
    ap.add_argument("--clear-radius", type=float, default=0.022,
                    help="tool-tip within this of the target counts as touchable (env default 0.022)")
    ap.add_argument("--gap-rad", type=float, default=0.5,
                    help="angular gap (rad) that separates two columns")
    ap.add_argument("--tilt-max", type=int, default=60,
                    help="max tool tilt (deg) about x and y searched per target")
    ap.add_argument("--tilt-step", type=int, default=20,
                    help="tilt grid step (deg); smaller = denser orientation search")
    ap.add_argument("--verbose", action="store_true", help="print every target")
    ap.add_argument("--json", default="", help="optional path to dump the summary as JSON")
    args = ap.parse_args()

    out = run_region(args.region, args.episodes, args.clear_radius, args.gap_rad,
                     args.verbose, args.tilt_max, args.tilt_step)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()

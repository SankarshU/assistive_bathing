import os
import numpy as np
import random
import pybullet as p
from gym import spaces

from .env import AssistiveEnv


class WipingEnv(AssistiveEnv):

    ENV_BUILD = "CONTINUITY-2026-06-11c (A:no-contact B:sticky-dist C:drift-term D:gap-gradient E:exit-penalty)"

    def __init__(self):
        super(WipingEnv, self).__init__()
        print(f"[WipingEnv] build: {self.ENV_BUILD} | file: {os.path.abspath(__file__)}")
        # auto_loop.py support: apply flag overrides from _env_overrides.json
        # (repo root) if present. Applied at the END of __init__ (see below) so
        # they win over the defaults set in this constructor.
        self._overrides_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "_env_overrides.json"
        )
        self.right_shoulder = 6
        self.right_elbow = 7
        self.right_wrist = 8
        self.human_controllable_joints = [3, 4, 5, 7]
        self.human_right_arm = [3, 4, 5, 6, 7, 8]

        # wiping task parameters (from assistive gym config)
        self.robot_forces = 1.0
        self.robot_gains = 0.05
        self.distance_weight = 1.0
        self.action_weight = 0.01
        self.wiping_reward_weight = 5.0
        self.task_success_threshold = 0.69

                # ---- reward shaping upgrades (paper-inspired) ----
        self.reach_switch_dist = 0.10     # meters; switch reach->wipe shaping
        self.near_target_dist = 0.06      # meters; dense proximity shaping
        self.contact_band_min = 2.0       # N  (start easier than 5N)
        self.contact_band_max = 12.0      # N
        self.force_weight = 0.02          # tune 0.01-0.05
        self.near_target_weight = 0.5     # tune 0.2-1.0
        self.clear_radius = 0.022 #0.035         # curriculum: start >0.025, tighten later

        # continuity (optional later)
        self.prev_contact_pos = None
        self.jump_thresh = 0.10
        self.jump_gamma = 0.3

        # ---- NEW reward upgrades (top picks from speaker-notes review) ----
        # Master switch for the 5 extended terms below:
        #   False -> documented 8-term reward (dist, act, clear, near, force, sweep, window, end)
        #   True  -> 13-term reward (8 documented + 5 extended)
        # Use this flag for the 8-vs-13 ablation.
        self.use_extended_reward = True
        # 1) Force-rate (skin-jerk) penalty: r_force_rate = -(F_t - F_{t-1})^2
        self.force_rate_weight = 0.005
        # 2) Tangential-velocity-in-contact reward (rewards real sliding)
        self.tang_vel_weight = 0.4
        self.tang_vel_cap = 0.05          # per-step cap on |Δalong-axis| (m)
        # 3) Sponge-orientation regulariser (sponge z-axis should be perpendicular to arm axis)
        self.orient_weight = 0.05
        # 4) In-band sustain bonus (small + per step while inside force band and wiping)
        self.sustain_weight = 0.05
        # 5) Staleness-weighted clears: late clears reward more (uses _phase_t as proxy age)
        self.staleness_alpha = 0.001
        self.staleness_age_cap = 2000     # cap so the bonus does not blow up

        # State for new rewards (lazy-initialised in step())
        self._prev_tool_pos = None        # for tangential velocity
        self._prev_normal_force = 0.0     # for force-rate penalty

        # ---- Anti-drift fixes (2026-06-11, from drift-away eval diagnosis) ----
        # Failure mode observed: policy clears the seeded-contact targets, loses
        # contact, exits wipe mode, and drifts away; the reach-phase pull
        # (-d_arm) plus tiny no-contact penalty (force_weight*-1 = -0.02/step)
        # is too weak to bring it back.
        # Fix A: dedicated no-contact penalty outside wipe mode (-w/step).
        self.no_contact_penalty_weight = 0.2    # 0.0 disables (ablation)
        # Fix B: sticky wipe-phase distance — after the first wipe-mode entry,
        # the distance reward keeps pulling toward the REMAINING TARGET CLOUD
        # even if wipe mode later exits (reach term only matters pre-contact).
        self.sticky_wipe_distance = True        # False disables (ablation)
        # Fix C: drift termination — end hopeless episodes early with a
        # terminal penalty, concentrating training data near the arm.
        self.drift_terminate = True             # False disables (ablation)
        self.drift_dist_thresh = 0.30           # meters
        self.drift_patience = 25                # consecutive steps beyond thresh
        self.drift_terminate_penalty = -15.0    # one-time penalty at termination

        # ---- Continuity fixes (2026-06-11c, from hover/poke-retreat diagnosis) ----
        # Observed: policy hovers ~5cm away (constant Fix-A penalty gives no
        # gradient to close the gap) and farms clears via touch-and-retreat.
        # Fix D: distance-SHAPED no-contact penalty — smoothly decreasing as the
        # tool approaches; gives the missing gradient over the final cm.
        self.gap_penalty_weight = 2.0           # 0.0 disables (ablation)
        # Fix E: wipe-mode exit penalty — each wipe->reach transition costs
        # reward, making poke-and-retreat strictly worse than staying in contact.
        self.wipe_exit_penalty = 2.0            # 0.0 disables (ablation)

        # Per-term weights for the directional terms (default 1.0 = current behavior;
        # set 0.0 to remove a term — enables the incremental "add one term at a time"
        # reward ablation starting from a bare distance+action+clears reward).
        self.sweep_weight = 1.0                 # 0.0 disables directional sweep reward
        self.window_weight = 1.0                # 0.0 disables local-window reward
        self.end_pass_weight = 1.0              # 0.0 disables end-of-pass bonus

        # ---- Fix F: anti-stall (2026-06-22, from s=1.0 end-of-arm stall diagnosis) ----
        # Observed: policy reaches the far end of the arm (s~1.0), stays in contact,
        # clears nothing, and parks there for the rest of the episode. No existing
        # penalty fires because Fix A/D only punish OUT-of-contact loitering. Penalize
        # (and optionally terminate) being in-contact with no clear for N steps while
        # targets remain — forces a sweep back to the remaining cloud. ALL DEFAULT OFF
        # so the frozen reward is byte-identical until explicitly enabled.
        self.stall_penalty_weight = 0.0         # >0 enables per-step in-contact-no-progress penalty
        self.stall_patience = 20                # steps in-contact w/o clear (targets remain) before stalled
        self.stall_terminate = False            # True ends the episode once stalled
        self.stall_terminate_penalty = -15.0    # one-time penalty at stall termination

        # Apply auto_loop.py overrides last so they beat all defaults above.
        try:
            import json as _json
            if os.path.exists(self._overrides_path):
                with open(self._overrides_path) as _f:
                    _ov = _json.load(_f)
                for _k, _v in _ov.items():
                    setattr(self, _k, _v)
                if _ov:
                    print(f"[WipingEnv] applied overrides from _env_overrides.json: {_ov}")
        except Exception as _e:
            print(f"[WipingEnv] override load failed (ignored): {_e}")

    def _enable_tool_human_collisions(self):
        # enable tool(-1,0,1) vs all right arm links
        tool_links = [-1, 0, 1]
        arm_links  = self.human_right_arm  # [3,4,5,6,7,8] (your list)
    
        for la in tool_links:
            for lb in arm_links:
                self.bc.setCollisionFilterPair(
                    bodyUniqueIdA=self.tool,
                    bodyUniqueIdB=self.humanoid._humanoid,
                    linkIndexA=la,
                    linkIndexB=lb,
                    enableCollision=1,
                    physicsClientId=self.bc._client
                )
    def _enable_tool_humanoid_collisions_all(self):
        tool_links = [-1] + list(range(self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client)))
        hum_links  = [-1] + list(range(self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)))
    
        for la in tool_links:
            for lb in hum_links:
                self.bc.setCollisionFilterPair(
                    self.tool, self.humanoid._humanoid,
                    la, lb,
                    enableCollision=1,
                    physicsClientId=self.bc._client
                )
    def _seed_tool_contact(self, max_iters=25, step_cap=0.01, target_gap=0.002):
        """
        Moves EEF a little bit toward right arm using IK until tool is within target_gap (meters).
        step_cap: max motion per iteration in meters (e.g., 1cm). Keeps it safe.
        """
        for _ in range(max_iters):
            # closest points tool link 0 -> right arm links
            cps = []
            for lb in self.human_right_arm:  # [3,4,5,6,7,8]
                cps.extend(self.bc.getClosestPoints(
                    bodyA=self.tool, bodyB=self.humanoid._humanoid,
                    distance=2.0,
                    linkIndexA=0, linkIndexB=lb,
                    physicsClientId=self.bc._client
                ))
            if not cps:
                return
    
            cmin = min(cps, key=lambda c: c[8])
            gap = float(cmin[8])
    
            # already close enough
            if gap <= target_gap:
                return
    
            pA = np.array(cmin[5])  # point on tool
            pB = np.array(cmin[6])  # point on human
            v = (pB - pA)
            n = np.linalg.norm(v) + 1e-9
            dir_to_arm = v / n
    
            # move only a small amount each iteration
            move = min(step_cap, gap - target_gap)
            disp = dir_to_arm * move
    
            eef_pos, eef_orn = self.bc.getLinkState(
                self.robot.id, self.robot.eef_id, computeForwardKinematics=True,
                physicsClientId=self.bc._client
            )[:2]
    
            new_eef_pos = (np.array(eef_pos) + disp).tolist()
    
            q = self.bc.calculateInverseKinematics(
                self.robot.id, self.robot.eef_id,
                new_eef_pos, eef_orn,
                physicsClientId=self.bc._client,
                maxNumIterations=80,
                residualThreshold=1e-4
            )
    
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.resetJointState(self.robot.id, joint_id, q[i], physicsClientId=self.bc._client)
    
            # settle a few steps each nudge
            for _ in range(5):
                self.bc.stepSimulation(physicsClientId=self.bc._client)

    

    def step(self, action):
        # 1) Apply action (advance physics + controller)
        if getattr(self, "_warmup_steps", 0) > 0:
            action = np.zeros_like(action)
            self._warmup_steps -= 1
        self.take_step(action, gains=self.robot_gains, forces=self.robot_forces)
    
        # ------------------------------------------------------------
        # Phase-stabilization state (lazy init: keep existing working logic)
        # ------------------------------------------------------------
        self._phase_t = getattr(self, "_phase_t", 0) + 1
        self._wipe_mode = getattr(self, "_wipe_mode", False)
        self._last_contact_t = getattr(self, "_last_contact_t", -10**9)
    
        # Added sweep-state (lazy init)
        self._prev_s = getattr(self, "_prev_s", None)
        self._pass_dir = getattr(self, "_pass_dir", 1.0)
        self._post_clear_grace = getattr(self, "_post_clear_grace", 0)
        self._arm_axis_cache = getattr(self, "_arm_axis_cache", None)
    
        # NEW: per-pass accumulated evidence for end-of-pass bonus
        self._pass_window_hits_accum = getattr(self, "_pass_window_hits_accum", 0)
        self._pass_window_clears_accum = getattr(self, "_pass_window_clears_accum", 0)

        # NEW: lazy-init state for the 5 new reward terms
        self._prev_tool_pos = getattr(self, "_prev_tool_pos", None)
        self._prev_normal_force = getattr(self, "_prev_normal_force", 0.0)

        # Tunable thresholds for hysteresis + latch
        enter_thresh = getattr(self, "wipe_enter_thresh", 0.010)
        exit_thresh = getattr(self, "wipe_exit_thresh", 0.030)
        latch_steps = getattr(self, "contact_latch_steps", 6)
        latch_force_eps = getattr(self, "latch_force_eps", 1e-6)
        soft_contact_dist = getattr(self, "soft_contact_dist", 0.010)
    
        # ------------------------------------------------------------
        # RAW CONTACT DIAGNOSTIC
        # ------------------------------------------------------------
        raw = self.bc.getContactPoints(
            bodyA=self.tool,
            bodyB=self.humanoid._humanoid,
            physicsClientId=self.bc._client
        )
        if raw is None:
            raw = []
        if not hasattr(self, "_dbg_raw_once"):
            self._dbg_raw_once = True
            # if len(raw) > 0:
            #     c = raw[0]
            #     print("RAW first contact:", "linkA=", c[3], "linkB=", c[4], "fn=", float(c[9]) if len(c) > 9 else None)
    
        if getattr(self, "_dbg_rawc", 0) % 2000 == 0:
            print("RAW contactPoints(tool-humanoid) =", len(raw))
        self._dbg_rawc = getattr(self, "_dbg_rawc", 0) + 1
    
        # ------------------------------------------------------------
        # DEBUG BLOCK (leave as-is)
        # ------------------------------------------------------------
        if not hasattr(self, "_dbg_tool_links"):
            self._dbg_tool_links = True
            n = self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client)
            for li in range(-1, n):
                cps = self.bc.getClosestPoints(
                    bodyA=self.tool,
                    bodyB=self.humanoid._humanoid,
                    distance=0.20,
                    linkIndexA=li,
                    physicsClientId=self.bc._client
                )
                mind = min([c[8] for c in cps], default=None)
                print(f"  tool link {li:2d}  minDist(within20cm)={mind}")
    
        if not hasattr(self, "_dbg_tool_collision"):
            self._dbg_tool_collision = True
            for li in [-1, 0, 1]:
                cs = self.bc.getCollisionShapeData(self.tool, li, physicsClientId=self.bc._client)
                print(f"collisionShapeData(tool, link={li}):", cs)
            vs = self.bc.getVisualShapeData(self.tool, physicsClientId=self.bc._client)
            print("visualShapeData(tool):", vs[:3], "... total=", len(vs))
    
        # ------------------------------------------------------------
        # 2) Compute wiping signals from filtered contacts
        # ------------------------------------------------------------
        clears, contact_pos, normal_force, near_target_contact = self.get_new_contact_points()
    
        if not hasattr(self, "_dbg_contactcount"):
            self._dbg_contactcount = 0
        self._dbg_contactcount += 1
        if self._dbg_contactcount % 2000 == 0:
            cps = self.bc.getContactPoints(
                bodyA=self.tool,
                bodyB=self.humanoid._humanoid,
                physicsClientId=self.bc._client
            )
            print("CONTACTPOINTS count(tool-human) =", len(cps))
    
        # ------------------------------------------------------------
        # 3) Observation after the step
        # ------------------------------------------------------------
        obs = self._get_obs()
    
        # ------------------------------------------------------------
        # 4) Distance-to-arm (phase-1): robust min over valid tool links (0,1)
        # ------------------------------------------------------------
        best = None
        for la in (0, 1):
            for lb in self.human_right_arm:
                cps = self.bc.getClosestPoints(
                    bodyA=self.tool,
                    bodyB=self.humanoid._humanoid,
                    distance=2.0,
                    linkIndexA=la,
                    linkIndexB=lb,
                    physicsClientId=self.bc._client
                )
                if cps:
                    d = min(c[8] for c in cps)
                    best = d if (best is None or d < best) else best
    
        if best is None:
            d_arm = 2.0
            reward_reach = -2.0
        else:
            d_arm = float(best)
            reward_reach = -d_arm
    
        # ------------------------------------------------------------
        # 4.5) Stable phase logic: hysteresis + soft-contact latch
        # ------------------------------------------------------------
        raw_contact_exists = (len(raw) > 0)
    
        # Soft contact event = forceful contact OR any raw contact OR ultra-near proximity
        contact_event = (
            (normal_force > latch_force_eps) or
            raw_contact_exists or
            (d_arm < soft_contact_dist)
        )
    
        if contact_event:
            self._last_contact_t = self._phase_t
    
        steps_since_contact = self._phase_t - self._last_contact_t
        contact_latched = (steps_since_contact <= latch_steps)
    
        wipe_exit_event = False  # Fix E: did wipe mode drop this step?
        if not self._wipe_mode:
            # Enter wipe mode when genuinely close to the arm
            if d_arm < enter_thresh:
                self._wipe_mode = True
                self._wipe_entered_once = True  # Fix B: episode-persistent
        else:
            # Exit only when clearly away and latch has expired
            if (d_arm > exit_thresh) and (not contact_latched):
                self._wipe_mode = False
                wipe_exit_event = True
                self._wipe_exits = getattr(self, "_wipe_exits", 0) + 1

        # Fix C: drift counter — consecutive steps far from the arm
        if d_arm > getattr(self, "drift_dist_thresh", 0.30):
            self._drift_steps = getattr(self, "_drift_steps", 0) + 1
        else:
            self._drift_steps = 0
    
        # ------------------------------------------------------------
        # 4.6) Diagnostics: near-arm but no registered contact
        # ------------------------------------------------------------
        near_1cm = (d_arm < 0.010)
        near_2cm = (d_arm < 0.020)
    
        # IMPORTANT: keep diagnostic "accepted" aligned with current latch logic
        accepted_contact_exists = contact_event
    
        self._diag_total_steps = getattr(self, "_diag_total_steps", 0) + 1
        self._diag_near_1cm = getattr(self, "_diag_near_1cm", 0)
        self._diag_near_2cm = getattr(self, "_diag_near_2cm", 0)
        self._diag_near1_no_raw = getattr(self, "_diag_near1_no_raw", 0)
        self._diag_near1_no_accept = getattr(self, "_diag_near1_no_accept", 0)
        self._diag_near2_no_raw = getattr(self, "_diag_near2_no_raw", 0)
        self._diag_near2_no_accept = getattr(self, "_diag_near2_no_accept", 0)
    
        if near_1cm:
            self._diag_near_1cm += 1
            if not raw_contact_exists:
                self._diag_near1_no_raw += 1
            if not accepted_contact_exists:
                self._diag_near1_no_accept += 1
    
        if near_2cm:
            self._diag_near_2cm += 1
            if not raw_contact_exists:
                self._diag_near2_no_raw += 1
            if not accepted_contact_exists:
                self._diag_near2_no_accept += 1
    
        if self._phase_t % 2000 == 0:
            print(
                "[DIAG] "
                f"steps={self._diag_total_steps} | "
                f"near<1cm={self._diag_near_1cm}, no_raw={self._diag_near1_no_raw}, no_accept={self._diag_near1_no_accept} | "
                f"near<2cm={self._diag_near_2cm}, no_raw={self._diag_near2_no_raw}, no_accept={self._diag_near2_no_accept}"
            )
    
        # Optional detailed probe for very-near / no-force cases
        if d_arm < 0.015 and normal_force <= 1e-6 and getattr(self, "_print_near_no_contact", False):
            print(f"[NEAR-NO-CONTACT] d_arm={d_arm:.4f}, raw_contacts={len(raw)}")
            for i, c in enumerate(raw[:5]):
                fn = float(c[9]) if len(c) > 9 else 0.0
                print(
                    f"  raw[{i}] linkA={c[3]} linkB={c[4]} dist={float(c[8]):.5f} fn={fn:.5f}"
                )
    
        # ------------------------------------------------------------
        # 5) Distance-to-targets (phase-2): coarse wipe shaping
        # ------------------------------------------------------------
        tool_pos_link0 = np.array(self.bc.getLinkState(
            self.tool, 0, computeForwardKinematics=True, physicsClientId=self.bc._client
        )[0])
    
        if hasattr(self, "feasible_targets_pos_world") and len(self.feasible_targets_pos_world) > 0:
            d_targets_mean = float(np.mean([
                np.linalg.norm(tool_pos_link0 - np.array(tw)) for tw in self.feasible_targets_pos_world
            ]))
            reward_wipe_dist = -d_targets_mean
        else:
            reward_wipe_dist = 0.0
    
        # Persistent phase switch (reach -> wipe)
        # Fix B: sticky wipe distance — after first wipe entry, keep pulling
        # toward the remaining target cloud even if wipe mode exits.
        self._wipe_entered_once = getattr(self, "_wipe_entered_once", False)
        use_wipe_dist = self._wipe_mode or (
            getattr(self, "sticky_wipe_distance", True) and self._wipe_entered_once
        )
        reward_distance = reward_wipe_dist if use_wipe_dist else reward_reach
    
        # ------------------------------------------------------------
        # 5.5) Arm-axis sweep shaping + NEW end-of-pass bonus
        # ------------------------------------------------------------
        reward_sweep = 0.0
        reward_end_pass = 0.0
        reward_cov = 0.0
        s = None
    
        # Use current contact point if available, else tool link-0 position
        tool_pos_for_s = np.array(contact_pos) if contact_pos is not None else tool_pos_link0
    
        # Build / refresh arm-axis from active target cloud for this arm_side
        active_world_targets = None
        if hasattr(self, "target_pos_world_dict") and hasattr(self, "arm_side"):
            active_world_targets = self.target_pos_world_dict.get(self.arm_side, None)
    
        if active_world_targets is not None and len(active_world_targets) >= 2:
            pts = np.asarray(active_world_targets, dtype=np.float32)
    
            # Recompute occasionally or on first use
            if (self._arm_axis_cache is None) or (self._phase_t % 200 == 1):
                center = np.mean(pts, axis=0)
                X = pts - center
                _, _, vh = np.linalg.svd(X, full_matrices=False)
                axis = vh[0]
    
                # Make axis direction stable over time
                if self._arm_axis_cache is not None:
                    old_axis = self._arm_axis_cache["axis"]
                    if np.dot(axis, old_axis) < 0:
                        axis = -axis
    
                proj = X @ axis
                s_min = float(np.min(proj))
                s_max = float(np.max(proj))
                span = max(s_max - s_min, 1e-6)

                # PC2 = the circumferential / column-separation axis (see analysis:
                # for this capsule geometry SVD's 2nd axis IS the "column" direction).
                # Cached for the optional (s, w) 2-D coverage term below.
                axis2 = vh[1] if vh.shape[0] > 1 else None
                if axis2 is not None:
                    if self._arm_axis_cache is not None and self._arm_axis_cache.get("axis2") is not None:
                        if np.dot(axis2, self._arm_axis_cache["axis2"]) < 0:
                            axis2 = -axis2
                    projw = X @ axis2
                    w_min = float(np.min(projw))
                    w_span = max(float(np.max(projw)) - w_min, 1e-6)
                else:
                    w_min, w_span = 0.0, 1.0

                self._arm_axis_cache = {
                    "center": center,
                    "axis": axis,
                    "s_min": s_min,
                    "s_max": s_max,
                    "span": span,
                    "axis2": axis2,
                    "w_min": w_min,
                    "w_span": w_span,
                }
    
            center = self._arm_axis_cache["center"]
            axis = self._arm_axis_cache["axis"]
            s_min = self._arm_axis_cache["s_min"]
            span = self._arm_axis_cache["span"]
    
            # Normalized position of tool along target-cloud axis
            s_raw = float(np.dot(tool_pos_for_s - center, axis))
            s = (s_raw - s_min) / span
            s = float(np.clip(s, 0.0, 1.0))

            # (s, w) 2-D COVERAGE shaping (flag-gated; default weight 0.0 -> no change).
            # Rewards the FIRST contact/clear in each not-yet-covered (length x width) cell,
            # where width w projects onto PC2 (the column axis). PC1-only sweep gives no
            # incentive to cover ACROSS columns; this closes that gap on the reachable arc.
            cov_w = getattr(self, "_pc2_coverage_weight", 0.0)
            if cov_w > 0.0 and self._arm_axis_cache.get("axis2") is not None \
                    and (near_target_contact or clears > 0):
                axis2 = self._arm_axis_cache["axis2"]
                w_min = self._arm_axis_cache["w_min"]
                w_span = self._arm_axis_cache["w_span"]
                w = float(np.clip((np.dot(tool_pos_for_s - center, axis2) - w_min) / w_span, 0.0, 1.0))
                nb = int(getattr(self, "_cov_bins", 4))
                cell = (min(nb - 1, int(s * nb)), min(nb - 1, int(w * nb)))
                if cell not in self._covered_cells:
                    self._covered_cells.add(cell)
                    reward_cov = 1.0

            if self._prev_s is None:
                self._prev_s = s
    
            # Short grace after a clear: don't punish tiny bounce too hard
            if clears > 0:
                self._post_clear_grace = 2
            else:
                self._post_clear_grace = max(0, self._post_clear_grace - 1)
    
            # Only shape sweep when actually wiping / contact-latched
            if self._wipe_mode or contact_latched:
                ds = s - self._prev_s
                ds_dir = self._pass_dir * ds
    
                eps = getattr(self, "sweep_eps", 0.01)          # deadband for jitter
                ds_cap = getattr(self, "sweep_ds_cap", 0.05)    # cap per-step effect
                k_fwd = getattr(self, "sweep_k_fwd", 0.5)       # keep shaping modest
                k_back = getattr(self, "sweep_k_back", 0.75)
                grace_scale = getattr(self, "sweep_grace_scale", 0.25)
    
                if ds_dir > eps:
                    reward_sweep = k_fwd * min(ds_dir, ds_cap)
                elif ds_dir < -eps:
                    back_amt = min(-ds_dir, ds_cap)
                    penalty_scale = grace_scale if self._post_clear_grace > 0 else 1.0
                    reward_sweep = -penalty_scale * k_back * back_amt
    
                # NEW: accumulate directional-window evidence during the current pass
                self._pass_window_hits_accum += int(getattr(self, "_window_hits", 0))
                self._pass_window_clears_accum += int(getattr(self, "_window_clears", 0))
    
                # Flip direction near the ends to encourage return pass
                # flip_hi = getattr(self, "sweep_flip_hi", 0.95)
                # flip_lo = getattr(self, "sweep_flip_lo", 0.05)

                flip_hi = getattr(self, "sweep_flip_hi", 0.90)
                flip_lo = getattr(self, "sweep_flip_lo", 0.10)
    
                # NEW: explicit end-of-pass bonus, but only if the pass had real wipe evidence
                min_pass_hits = getattr(self, "pass_bonus_min_hits", 1)
                min_pass_clears = getattr(self, "pass_bonus_min_clears", 1)
                end_pass_bonus = getattr(self, "end_pass_reward", 4.0)
    
                reached_forward_end = (self._pass_dir > 0 and s >= flip_hi)
                reached_backward_end = (self._pass_dir < 0 and s <= flip_lo)
                pass_has_progress = (
                    (self._pass_window_hits_accum >= min_pass_hits) or
                    (self._pass_window_clears_accum >= min_pass_clears)
                )
    
                if pass_has_progress and (reached_forward_end or reached_backward_end):
                    reward_end_pass = end_pass_bonus
    
                # After evaluating bonus, flip direction and reset per-pass accumulators
                if reached_forward_end:
                    self._pass_dir = -1.0
                    self._pass_window_hits_accum = 0
                    self._pass_window_clears_accum = 0
                elif reached_backward_end:
                    self._pass_dir = 1.0
                    self._pass_window_hits_accum = 0
                    self._pass_window_clears_accum = 0
            else:
                # NEW: if we are clearly not in an active wiping phase, reset pass evidence
                self._pass_window_hits_accum = 0
                self._pass_window_clears_accum = 0
    
            self._prev_s = s
    
        # ------------------------------------------------------------
        # 6) Action penalty
        # ------------------------------------------------------------
        reward_action = -float(np.sum(np.square(action)))
    
        # 7) Wipe progress reward (sparse clears)
        reward_clears = float(clears)
    
        # 8) Dense proxy reward: any contact near remaining targets
        reward_near_target = float(near_target_contact)
    
        # Prefer contact / clears in the active directional wipe window
        reward_window = 0.0
        if self._wipe_mode and contact_latched:
            reward_window += 0.5 * float(getattr(self, "_window_near_target_contact", 0))
            reward_window += 0.25 * float(getattr(self, "_window_clears", 0))
    
        # 9) Force-band shaping (encourage safe sustained contact)
        if normal_force <= 1e-6:
            reward_force = 0.0 if self._wipe_mode else -1.0
        else:
            if normal_force < self.contact_band_min:
                reward_force = -(self.contact_band_min - normal_force)
            elif normal_force > self.contact_band_max:
                reward_force = -(normal_force - self.contact_band_max)
            else:
                reward_force = 0.0

        # Fix A: dedicated no-contact penalty outside wipe mode.
        # (The force-band path above only contributes force_weight*-1 = -0.02/step,
        #  which made drifting away nearly free.)
        reward_no_contact = -1.0 if (normal_force <= 1e-6 and not self._wipe_mode) else 0.0

        # Fix D: distance-shaped gap penalty whenever there is no actual contact
        # force. Unlike Fix A (constant), this decreases smoothly as the tool
        # approaches, providing the gradient to close the final centimeters.
        reward_gap = -min(d_arm, 0.30) if normal_force <= 1e-6 else 0.0

        # Fix E: one-time penalty when wipe mode drops (poke-and-retreat deterrent)
        reward_wipe_exit = -1.0 if wipe_exit_event else 0.0
    
        # ------------------------------------------------------------
        # 9.5) EXTENDED reward terms (force-rate, tangential vel, orient, sustain, staleness)
        #      Gated by self.use_extended_reward (8-term vs 13-term ablation flag).
        # ------------------------------------------------------------
        if getattr(self, "use_extended_reward", True):
            # 9.5.a) Force-rate penalty: penalise large |ΔF| step-to-step
            df = float(normal_force) - float(self._prev_normal_force)
            reward_force_rate = -(df * df)

            # 9.5.b) Tangential-velocity-in-contact reward
            # Project the per-step tool displacement onto the cached arm axis; reward only while
            # actually pressing on the arm. Capped per-step to avoid huge spikes.
            reward_tang = 0.0
            if (
                (normal_force > 1e-6)
                and (self._prev_tool_pos is not None)
                and (self._arm_axis_cache is not None)
            ):
                axis = self._arm_axis_cache["axis"]
                delta = np.asarray(tool_pos_link0, dtype=np.float32) - np.asarray(self._prev_tool_pos, dtype=np.float32)
                tang_speed = float(abs(np.dot(delta, axis)))
                reward_tang = float(min(tang_speed, self.tang_vel_cap))

            # 9.5.c) Sponge-orientation regulariser
            # The sponge body z-axis (in world) should be roughly perpendicular to the arm axis,
            # i.e. its dot-product with the arm tangent should be small. Penalise edge-on contact.
            reward_orient = 0.0
            if self._wipe_mode and (self._arm_axis_cache is not None):
                try:
                    _, tool_orn1 = self.bc.getLinkState(
                        self.tool, 1, computeForwardKinematics=True,
                        physicsClientId=self.bc._client
                    )[:2]
                    Rm = self.bc.getMatrixFromQuaternion(tool_orn1)
                    # Tool z-axis in world (column 2 of rotation matrix, row-major 9-tuple)
                    z_tool = np.array([Rm[2], Rm[5], Rm[8]], dtype=np.float32)
                    axis_arm = np.asarray(self._arm_axis_cache["axis"], dtype=np.float32)
                    proj = float(np.dot(z_tool, axis_arm))
                    reward_orient = -(proj * proj)
                except Exception:
                    reward_orient = 0.0

            # 9.5.d) In-band sustain bonus (small +reward per step inside [band_min, band_max])
            if self._wipe_mode and (self.contact_band_min <= normal_force <= self.contact_band_max):
                reward_sustain = 1.0
            else:
                reward_sustain = 0.0

            # 9.5.e) Staleness-weighted clear bonus
            # Episode-time proxy for "how long it has been hard to clear targets". Late clears
            # therefore receive an extra bonus, encouraging the policy to mop up the last 30%.
            age_proxy = float(min(self._phase_t, self.staleness_age_cap))
            reward_stale = self.staleness_alpha * age_proxy * float(clears)
        else:
            # 8-term documented reward: extended terms contribute exactly zero.
            reward_force_rate = 0.0
            reward_tang = 0.0
            reward_orient = 0.0
            reward_sustain = 0.0
            reward_stale = 0.0

        # ------------------------------------------------------------
        # Fix F: anti-stall bookkeeping. Count consecutive in-contact steps with
        # no clear while targets remain; reset on any clear. Used for the stall
        # penalty + optional termination below. (All gated; off by default.)
        # ------------------------------------------------------------
        _targets_remain = (len(self.feasible_targets_pos_world) > 0) \
            if hasattr(self, "feasible_targets_pos_world") else True
        _in_contact_or_wipe = bool(accepted_contact_exists) or bool(self._wipe_mode)
        if clears > 0:
            self._steps_since_clear = 0
        elif _in_contact_or_wipe and _targets_remain:
            self._steps_since_clear = getattr(self, "_steps_since_clear", 0) + 1
        else:
            self._steps_since_clear = getattr(self, "_steps_since_clear", 0)
        _stalled = self._steps_since_clear > getattr(self, "stall_patience", 20)
        reward_stall = -1.0 if _stalled else 0.0

        # ------------------------------------------------------------
        # 10) Total reward
        # ------------------------------------------------------------
        if getattr(self, "use_baseline_reward", False):
            # ---- ORIGINAL RL BASELINE (yubink2/learn-bathing, git 96fb9af) ----
            # 3-term contact-counting reward:  r = 1.0*(-d) + 0.01*(-||a||^2) + 5.0*contacts
            # Fully gated: default off => the structured reward below is byte-identical.
            # Faithfulness: uses reward_reach (-d_arm) for the distance term and `clears`
            # as the contact-count analogue. For a fully faithful run also set
            # clear_radius=0.025 and task_success_threshold=0.6 via overrides, OR train
            # directly on git 96fb9af (see BASELINES_PLAN.md).
            # SCOPE (be precise in the paper): only the reward TOTAL differs in this
            # branch — none of the sweep/window/end/extended REWARD terms are paid.
            # Episode MECHANICS applied outside this if/else still apply to yubik runs:
            # drift termination (Fix C, incl. its -15 terminal penalty) and
            # wipe_start_init. That is the intended isolation (identical episode
            # protocol for all methods; only the reward differs) and is confirmed in
            # the data (Tyubik drift-termination rates up to ~0.3). Describe the
            # baseline as "contact-counting reward under our episode protocol",
            # NOT as a byte-identical reproduction of the original training setup.
            reward = 1.0 * reward_reach + 0.01 * reward_action + 5.0 * float(clears)
        else:
            reward = (
                self.distance_weight * reward_distance +
                self.action_weight * reward_action +
                self.wiping_reward_weight * reward_clears +
                self.near_target_weight * reward_near_target +
                self.force_weight * reward_force +
                getattr(self, "no_contact_penalty_weight", 0.2) * reward_no_contact +  # Fix A
                getattr(self, "gap_penalty_weight", 2.0) * reward_gap +                # Fix D
                getattr(self, "wipe_exit_penalty", 2.0) * reward_wipe_exit +           # Fix E
                getattr(self, "sweep_weight", 1.0) * reward_sweep +
                getattr(self, "window_weight", 1.0) * reward_window +
                getattr(self, "end_pass_weight", 1.0) * reward_end_pass +
                getattr(self, "_pc2_coverage_weight", 0.0) * reward_cov +      # (s,PC2) 2-D coverage
                # Extended terms (all exactly 0.0 when use_extended_reward=False)
                self.force_rate_weight * reward_force_rate +
                self.tang_vel_weight   * reward_tang +
                self.orient_weight     * reward_orient +
                self.sustain_weight    * reward_sustain +
                reward_stale +
                getattr(self, "stall_penalty_weight", 0.0) * reward_stall   # Fix F
            )

        # Fix C: drift termination — apply terminal penalty and flag done
        terminated_drift = bool(
            getattr(self, "drift_terminate", True)
            and self._drift_steps >= getattr(self, "drift_patience", 25)
        )
        if terminated_drift:
            reward += getattr(self, "drift_terminate_penalty", -15.0)

        # Fix F: stall termination — optionally end episodes parked in-contact
        terminated_stall = bool(
            getattr(self, "stall_terminate", False)
            and self._steps_since_clear >= getattr(self, "stall_patience", 20)
        )
        if terminated_stall:
            reward += getattr(self, "stall_terminate_penalty", -15.0)

        # NEW: update prev-state for next step (do this AFTER reward is computed)
        self._prev_normal_force = float(normal_force)
        self._prev_tool_pos = np.array(tool_pos_link0, dtype=np.float32).copy()
    
        # ------------------------------------------------------------
        # 11) Info + termination
        # ------------------------------------------------------------
        info = {
            'number_of_targets_cleared': self.task_success,
            'total_target_count': self.total_target_count,
            'feasible_target_count': self.feasible_targets_count,
            'task_success': int(self.task_success >= (self.feasible_targets_count * self.task_success_threshold)),
            'task_percentage': self.task_success / self.feasible_targets_count * 100,
            'action_robot_len': self.action_robot_len,
            'obs_robot_len': self.obs_robot_len,
    
            "d_arm": float(d_arm),
            "normal_force": float(normal_force),
            "near_target_contact": int(near_target_contact),
            "clears_this_step": int(clears),
            "n_remaining_targets": int(len(self.feasible_targets_pos_world)) if hasattr(self, "feasible_targets_pos_world") else 0,
    
            "wipe_mode": int(self._wipe_mode),
            "steps_since_contact": int(steps_since_contact),
            "contact_latched": int(contact_latched),
    
            "raw_contact_exists": int(raw_contact_exists),
            "accepted_contact_exists": int(accepted_contact_exists),
            "near_1cm": int(near_1cm),
            "near_2cm": int(near_2cm),
    
            # Sweep diagnostics
            "s": float(s) if s is not None else -1.0,
            "pass_dir": float(self._pass_dir),
            "reward_sweep": float(reward_sweep),
            "reward_end_pass": float(reward_end_pass),
            "post_clear_grace": int(self._post_clear_grace),
    
            # Window diagnostics
            "window_near_target_contact": int(getattr(self, "_window_near_target_contact", 0)),
            "window_hits": int(getattr(self, "_window_hits", 0)),
            "window_clears": int(getattr(self, "_window_clears", 0)),
    
            # NEW: per-pass accumulated evidence
            "pass_window_hits_accum": int(self._pass_window_hits_accum),
            "pass_window_clears_accum": int(self._pass_window_clears_accum),

            # Anti-drift fix diagnostics (Fixes A/B/C + continuity D/E)
            "reward_gap": float(reward_gap),
            "reward_wipe_exit": float(reward_wipe_exit),
            "wipe_exits": int(getattr(self, "_wipe_exits", 0)),
            "reward_no_contact": float(reward_no_contact),
            "wipe_entered_once": int(self._wipe_entered_once),
            "sticky_wipe_dist_active": int(use_wipe_dist and not self._wipe_mode),
            "drift_steps": int(self._drift_steps),
            "terminated_drift": int(terminated_drift),
            "steps_since_clear": int(self._steps_since_clear),   # Fix F
            "reward_stall": float(reward_stall),                 # Fix F
            "terminated_stall": int(terminated_stall),           # Fix F

            # Extended reward-component diagnostics (all 0.0 when use_extended_reward=False)
            "use_extended_reward": int(getattr(self, "use_extended_reward", True)),
            "reward_force_rate": float(reward_force_rate),
            "reward_tang":       float(reward_tang),
            "reward_orient":     float(reward_orient),
            "reward_sustain":    float(reward_sustain),
            "reward_stale":      float(reward_stale),
        }
    
        done = bool(self.task_success >= (self.feasible_targets_count * self.task_success_threshold)) \
            or terminated_drift or terminated_stall
        if terminated_drift:
            print(f"[ANTIDRIFT] episode terminated: d_arm={d_arm:.3f} > "
                  f"{getattr(self, 'drift_dist_thresh', 0.30):.2f} for {self._drift_steps} steps "
                  f"(penalty {getattr(self, 'drift_terminate_penalty', -15.0):+.1f})")
        if terminated_stall:
            print(f"[ANTISTALL] episode terminated: in-contact no-clear for "
                  f"{self._steps_since_clear} steps (penalty "
                  f"{getattr(self, 'stall_terminate_penalty', -15.0):+.1f})")
    
        # ------------------------------------------------------------
        # Debug print
        # ------------------------------------------------------------
        self._dbg_t = getattr(self, "_dbg_t", 0) + 1
        if self._dbg_t % 500 == 0:
            n_t = len(self.feasible_targets_pos_world) if hasattr(self, "feasible_targets_pos_world") else 0
            s_dbg = -1.0 if s is None else s
            print(
                f"*d_arm={d_arm:.3f}  force={normal_force:.4f}  raw={int(raw_contact_exists)}  "
                f"accept={int(accepted_contact_exists)}  near={near_target_contact}  clears={clears}  "
                f"n_targets={n_t}  wipe_mode={int(self._wipe_mode)}  "
                f"latched={int(contact_latched)}  steps_since_contact={steps_since_contact}  "
                f"s={s_dbg:.3f}  dir={self._pass_dir:+.0f}  rsweep={reward_sweep:.3f}  "
                f"rend={reward_end_pass:.3f}  "
                f"wh={int(getattr(self, '_window_hits', 0))}  "
                f"wc={int(getattr(self, '_window_clears', 0))}  "
                f"pwh={int(self._pass_window_hits_accum)}  "
                f"pwc={int(self._pass_window_clears_accum)}"
            )
    
        return obs, reward, done, info

    def get_new_contact_points(self):
        """
        Returns:
          clears: int (#unique targets cleared this step)
          contact_pos: np.array or None
          normal_force: float
          near_target_contact: int
    
        Added side effects (for Stage 2A shaping only):
          self._window_near_target_contact
          self._window_hits
          self._window_clears
        """
        clears = 0
        contact_pos = None
        normal_force = 0.0
        near_target_contact = 0
    
        # New: directional-window diagnostics/shaping signals
        self._window_near_target_contact = 0
        self._window_hits = 0
        self._window_clears = 0
    
        # --- fetch contacts once ---
        cps = self.bc.getContactPoints(
            bodyA=self.tool,
            bodyB=self.humanoid._humanoid,
            physicsClientId=self.bc._client
        )
        if cps is None:
            cps = []
        if not hasattr(self, "_dbg_contact_once"):
            self._dbg_contact_once = True
            print("RAW cps:", len(cps))
            if len(cps) > 0:
                c0 = cps[0]
                print("RAW first:", "linkA=", c0[3], "linkB=", c0[4], "fn=", float(c0[9]))
    
        def _delete_indices(obj, idxs):
            if obj is None:
                return None
            if isinstance(obj, list):
                for i in idxs:
                    del obj[i]
                return obj
            try:
                return np.delete(obj, idxs, axis=0)
            except Exception:
                return np.delete(obj, idxs)
    
        def _len(obj):
            if obj is None:
                return 0
            return len(obj)
    
        # ------------------------------------------------------------
        # Build active directional window over current feasible targets
        # ------------------------------------------------------------
        window_mask = None
        targets_s = None
    
        if hasattr(self, "feasible_targets_pos_world") and _len(self.feasible_targets_pos_world) > 0:
            targets_world_full = np.asarray(self.feasible_targets_pos_world, dtype=np.float32)
    
            axis_cache = getattr(self, "_arm_axis_cache", None)
            pass_dir = float(getattr(self, "_pass_dir", 1.0))
            prev_s = getattr(self, "_prev_s", None)
    
            # Need valid axis and current s to define directional window
            if axis_cache is not None and prev_s is not None:
                center = axis_cache["center"]
                axis = axis_cache["axis"]
                s_min = axis_cache["s_min"]
                span = axis_cache["span"]
    
                proj = (targets_world_full - center) @ axis
                targets_s = (proj - s_min) / span
                targets_s = np.clip(targets_s, 0.0, 1.0)
    
                win_half = float(getattr(self, "sweep_window_size", 0.12))
    
                if pass_dir > 0:
                    lo, hi = prev_s, prev_s + win_half
                else:
                    lo, hi = prev_s - win_half, prev_s
    
                lo = max(0.0, lo)
                hi = min(1.0, hi)
    
                window_mask = (targets_s >= lo) & (targets_s <= hi)
    
        # Early exit: no targets left
        if (not hasattr(self, "feasible_targets_pos_world")) or (_len(self.feasible_targets_pos_world) == 0):
            for c in cps:
                linkA = c[3]
                linkB = c[4]
    
                if linkA not in (0, 1):
                    continue
                if linkB < 0 or linkB not in self.human_right_arm:
                    continue
    
                fn = float(c[9]) if len(c) > 9 else 0.0
                normal_force += fn
                contact_pos = np.asarray(c[6], dtype=np.float32)
    
            return 0, (None if contact_pos is None else np.array(contact_pos)), float(normal_force), 0
    
        targets_world = np.asarray(self.feasible_targets_pos_world, dtype=np.float32)
        to_clear = set()
        to_clear_in_window = set()
    
        min_clear_force = float(getattr(self, "min_clear_force", 0.0))
    
        for c in cps:
            linkA = c[3]
            linkB = c[4]
    
            if linkA not in (0, 1):
                continue
            if linkB < 0 or linkB not in self.human_right_arm:
                continue
    
            posB = np.asarray(c[6], dtype=np.float32)
            fn = float(c[9]) if len(c) > 9 else 0.0
    
            normal_force += fn
            contact_pos = posB
    
            # distances from this contact point to all remaining targets
            dists = np.linalg.norm(targets_world - posB[None, :], axis=1)
    
            if np.min(dists) < self.near_target_dist:
                near_target_contact = 1
    
            # New: directional-window-aware contact signal
            if window_mask is not None and np.any(window_mask):
                window_dists = dists[window_mask]
    
                if len(window_dists) > 0 and np.min(window_dists) < self.near_target_dist:
                    self._window_near_target_contact = 1
    
                self._window_hits += int(np.sum(window_dists < self.near_target_dist))
    
            # Old clear logic remains unchanged
            if fn >= min_clear_force:
                hit_idx = np.where(dists < self.clear_radius)[0]
                for i in hit_idx.tolist():
                    to_clear.add(int(i))
    
                    # New: record whether this clear happened in the active window
                    if window_mask is not None and i < len(window_mask) and window_mask[i]:
                        to_clear_in_window.add(int(i))
    
        # Apply clears (unchanged)
        if to_clear:
            idxs = sorted(to_clear, reverse=True)
    
            for i in idxs:
                target_body = self.feasible_targets[i]
                self.bc.resetBasePositionAndOrientation(
                    target_body, [1000, 1000, 1000], [0, 0, 0, 1],
                    physicsClientId=self.bc._client
                )
    
            self.feasible_targets = _delete_indices(self.feasible_targets, idxs)
            self.feasible_targets_pos_world = _delete_indices(self.feasible_targets_pos_world, idxs)
            if hasattr(self, "feasible_targets_pos") and self.feasible_targets_pos is not None:
                self.feasible_targets_pos = _delete_indices(self.feasible_targets_pos, idxs)
    
            clears = len(idxs)
            self.task_success += clears
    
        # New: how many clears happened in the desired directional window
        self._window_clears = len(to_clear_in_window)
    
        return (
            clears,
            (None if contact_pos is None else np.array(contact_pos)),
            float(normal_force),
            int(near_target_contact),
        )
    
    def _get_obs(self):
        obs = np.ones(self.obs_robot_len) * 30  # padding irrelevant obs values to 30
        real_obs_len = 3 + 4 + 6 + len(self.feasible_targets_pos_world) * 3
    
        tool_pos, tool_orn = self.bc.getLinkState(
            self.tool, 1, computeForwardKinematics=True, physicsClientId=self.bc._client
        )[:2]
        tool_pos = np.array(tool_pos)
        tool_orn = np.array(tool_orn)
    
        robot_joint_states = self.bc.getJointStates(
            self.robot.id,
            jointIndices=self.robot.arm_controllable_joints,
            physicsClientId=self.bc._client
        )
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])
    
        real_obs = np.concatenate(
            [tool_pos, tool_orn, robot_joint_positions] + list(self.feasible_targets_pos_world)
        ).ravel().astype(np.float32)
    
        # print("feasible_targets_count =", self.feasible_targets_count)
        # print("obs_robot_len =", self.obs_robot_len)
        # print("real_obs_len =", real_obs_len)
    
        assert real_obs_len <= self.obs_robot_len, (
            f"Observation overflow: real_obs_len={real_obs_len}, obs_robot_len={self.obs_robot_len}"
        )
    
        obs[:real_obs_len] = real_obs
        return obs

    def _apply_pose_roll(self):
        """Pose-configuration hook (the 'pose' context axis for multi-config coverage).

        Applies an extra roll of `_pose_roll_delta` radians to human joint `_pose_roll_joint`
        AFTER the base arm config is set/locked and BEFORE targets are generated, so the
        target cloud is placed on the re-posed limb. This exposes the band that is otherwise
        occluded against the bed in the nominal supine pose (a single fixed pose cannot cover
        the full limb circumference). Both fields default to a no-op, and may be set as an
        attribute or via _env_overrides.json (same mechanism as the region override), so
        existing single-pose results are byte-identical unless a pose is requested.

        Returns True if the resulting pose is collision-free (or no roll was requested),
        False if it self-collides / hits the bed (caller resamples the human config).
        """
        # A pose can be a single roll (_pose_roll_joint/_pose_roll_delta) OR a combined
        # posture (_pose_joint_deltas = {joint: delta, ...}, e.g. lift + roll + yaw).
        deltas = dict(getattr(self, "_pose_joint_deltas", None) or {})
        if not deltas:
            j = getattr(self, "_pose_roll_joint", None)
            d = float(getattr(self, "_pose_roll_delta", 0.0) or 0.0)
            if j is not None and d != 0.0:
                deltas = {int(j): d}
        if not deltas:
            return True
        for joint, delta in deltas.items():
            joint, delta = int(joint), float(delta)
            if delta == 0.0:
                continue
            cur = self.bc.getJointState(self.humanoid._humanoid, joint,
                                        physicsClientId=self.bc._client)[0]
            newq = cur + delta
            self.bc.resetJointState(self.humanoid._humanoid, joint, newq,
                                    physicsClientId=self.bc._client)
            # hold the re-posed joint through the episode (mirrors lock_human_joints)
            self.bc.setJointMotorControl2(self.humanoid._humanoid, joint,
                                          controlMode=p.POSITION_CONTROL, targetPosition=newq,
                                          force=1000, physicsClientId=self.bc._client)
        self.bc.stepSimulation(physicsClientId=self.bc._client)
        return not self.human_in_collision()

    def _reachable_target_mask(self, world_pts, local_pts):
        """Boolean keep-mask over a region's targets: keep columns whose mean outward surface
        normal faces the robot, drop columns averted from it (bed-facing underside). Columns
        are found by gap-clustering the LOCAL azimuth; facing is normal · direction-to-robot.
        Threshold `_reachable_facing_thresh` (default -0.15) is the cosine below which a column
        is judged unwipeable. Never drops every column."""
        W = np.asarray(world_pts, dtype=float)
        if len(W) < 3:
            return np.ones(len(W), dtype=bool)
        c = W.mean(axis=0)
        _, _, Vt = np.linalg.svd(W - c, full_matrices=False)
        axis = Vt[0]
        robot = np.asarray(self.robot_base_pose[0], dtype=float)
        L = np.asarray(local_pts, dtype=float)
        phi = np.arctan2(L[:, 1], L[:, 0])
        cmean = np.arctan2(np.sin(phi).mean(), np.cos(phi).mean())
        a = (phi - cmean + np.pi) % (2 * np.pi) - np.pi
        order = np.argsort(a)
        cols = np.empty(len(a), dtype=int)
        k = 0
        cols[order[0]] = 0
        for i in range(1, len(order)):
            if a[order[i]] - a[order[i - 1]] > 0.5:
                k += 1
            cols[order[i]] = k
        thresh = float(getattr(self, "_reachable_facing_thresh", -0.15))
        keep = np.ones(len(W), dtype=bool)
        for cc in set(cols.tolist()):
            idx = np.where(cols == cc)[0]
            P = W[idx]
            normal = P - (c + ((P - c) @ axis)[:, None] * axis)
            n = normal.mean(axis=0)
            n = n / (np.linalg.norm(n) + 1e-9)
            tor = robot - P.mean(axis=0)
            tor = tor / (np.linalg.norm(tor) + 1e-9)
            if float(n @ tor) < thresh:
                keep[idx] = False
        if not keep.any():          # safety: never scope the region to nothing
            keep[:] = True
        return keep

    def reset(self):
        self._warmup_steps = 30
        self.bc.resetSimulation(self.bc._client)
        self.setup_timing()
        self.create_world()
    
        self.task_success = 0
        self.contact_points_on_arm = {}
        self.robot_lower_limits = self.robot.arm_lower_limits
        self.robot_upper_limits = self.robot.arm_upper_limits

        # ✅ FIX: explicitly reset per-episode reward/phase state.
        # Previously these were lazily initialized via getattr() in step(),
        # so wipe mode, pass direction, arm-axis cache, latch timing, and
        # pass-evidence accumulators leaked across episodes.
        self._phase_t = 0
        self._wipe_mode = False
        self._last_contact_t = -10**9
        self._prev_s = None
        self._pass_dir = 1.0
        self._post_clear_grace = 0
        self._arm_axis_cache = None
        self._covered_cells = set()          # (s,w) cells covered this episode (PC2 coverage term)
        self._pass_window_hits_accum = 0
        self._pass_window_clears_accum = 0
        self._window_hits = 0
        self._window_clears = 0
        self._window_near_target_contact = 0
        self._prev_tool_pos = None
        self._prev_normal_force = 0.0
        self.prev_contact_pos = None

        # Anti-drift fix state (Fixes B/C) + continuity (E)
        self._wipe_entered_once = False
        self._drift_steps = 0
        self._wipe_exits = 0
        self._steps_since_clear = 0      # Fix F: anti-stall counter (reset per episode)

        # One-line config log every reset: confirms which build/fixes are live
        print(
            f"[WipingEnv reset] build={self.ENV_BUILD.split(' ')[0]} | "
            f"extended_reward={getattr(self, 'use_extended_reward', True)} | "
            f"A:no_contact_w={getattr(self, 'no_contact_penalty_weight', 0.2)} | "
            f"B:sticky_wipe={getattr(self, 'sticky_wipe_distance', True)} | "
            f"C:drift_term={getattr(self, 'drift_terminate', True)} "
            f"(>{getattr(self, 'drift_dist_thresh', 0.30)}m x{getattr(self, 'drift_patience', 25)}, "
            f"pen={getattr(self, 'drift_terminate_penalty', -15.0)}) | "
            f"D:gap_w={getattr(self, 'gap_penalty_weight', 2.0)} | "
            f"E:exit_pen={getattr(self, 'wipe_exit_penalty', 2.0)} | "
            f"region={getattr(self, '_target_trial_order_override', ['forearm_back'])}"
        )
        if getattr(self, "_reachable_only", False):
            print(f"[WipingEnv reset] reachable-arc mode ON "
                  f"(facing_thresh={getattr(self, '_reachable_facing_thresh', -0.15)})")
        if getattr(self, "_pc2_coverage_weight", 0.0):
            print(f"[WipingEnv reset] (s,PC2) coverage term ON "
                  f"(weight={float(self._pc2_coverage_weight)}, bins={getattr(self, '_cov_bins', 4)})")
        if getattr(self, "_pose_range_override", None):
            print(f"[WipingEnv reset] pose-bin active: q_H range override {self._pose_range_override}")
        if getattr(self, "_pose_joint_deltas", None):
            print(f"[WipingEnv reset] pose active (combined): {self._pose_joint_deltas}")
        elif getattr(self, "_pose_roll_delta", 0.0):
            print(f"[WipingEnv reset] pose-roll active: joint={getattr(self, '_pose_roll_joint', None)} "
                  f"delta={float(self._pose_roll_delta):+.3f} rad")

        feasible_targets_found = False
        while True:
            # randomize human arm config and check for collisions
            self.robot.reset()
            self.reset_and_check()

            # Pose-configuration hook: optionally re-pose (roll) the limb so an otherwise
            # bed-occluded band is exposed. No-op unless _pose_roll_delta is set. If the
            # rolled pose self-collides / hits the bed, resample the human config.
            if not self._apply_pose_roll():
                continue

            feasible_targets_found = self.generate_targets()
            if not feasible_targets_found:
                self.remove_targets()
                continue
    
            # ✅ FIX: make targets mutable lists (prevents "cannot delete array elements")
            self.feasible_targets = list(self.feasible_targets)
            self.feasible_targets_pos_world = list(self.feasible_targets_pos_world)
            if hasattr(self, "feasible_targets_pos") and self.feasible_targets_pos is not None:
                self.feasible_targets_pos = list(self.feasible_targets_pos)
    
            if self.gui:
                self.mark_feasible_targets()
    
            # initialize tool & compute eef_to_tool
            self.robot.reset()
            self.init_tool()
    
            # reset robot to init config
            for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                self.bc.resetJointState(self.robot.id, joint_id, self.init_q_R[i], physicsClientId=self.bc._client)
    
            # reset tool and attach it to eef
            world_to_eef = self.bc.getLinkState(
                self.robot.id, self.robot.eef_id, computeForwardKinematics=True,
                physicsClientId=self.bc._client
            )[:2]
    
            world_to_tool = self.bc.multiplyTransforms(
                world_to_eef[0], world_to_eef[1],
                self.eef_to_tool[0], self.eef_to_tool[1],
                physicsClientId=self.bc._client
            )
    
            self.bc.resetBasePositionAndOrientation(
                self.tool, world_to_tool[0], world_to_tool[1],
                physicsClientId=self.bc._client
            )
            self.attach_tool()
    
            break
    
        self._enable_tool_humanoid_collisions_all()
        self._seed_tool_contact(max_iters=25, step_cap=0.01, target_gap=0.002)

        # Fix D: wipe-start initialization (legitimate replacement for what the
        # old cross-episode leak provided by accident). The tool was just seeded
        # to ~2 mm from the arm — inside the wipe-enter threshold — so starting
        # in wipe mode with a fresh contact latch is honest, and it makes every
        # episode generate wipe-phase data from step 1. This keeps the sweep /
        # window / end-of-pass machinery active during training instead of
        # dormant until the policy independently masters reach.
        if getattr(self, "wipe_start_init", True):
            d0 = None
            try:
                cps0 = []
                for lb in self.human_right_arm:
                    cps0.extend(self.bc.getClosestPoints(
                        bodyA=self.tool, bodyB=self.humanoid._humanoid,
                        distance=0.5, linkIndexA=0, linkIndexB=lb,
                        physicsClientId=self.bc._client))
                if cps0:
                    d0 = min(float(c[8]) for c in cps0)
            except Exception:
                d0 = None
            enter0 = getattr(self, "wipe_enter_thresh", 0.010)
            if d0 is not None and d0 < enter0:
                self._wipe_mode = True
                self._wipe_entered_once = True
                self._last_contact_t = 0   # latch fresh at episode start
                print(f"[WipingEnv reset] Fix D: wipe-start init active (seeded d_arm={d0:.4f} < {enter0})")
            else:
                print(f"[WipingEnv reset] Fix D: seeding missed (d_arm={d0}), starting in reach phase")

        # NEW: reset state for the 5 new reward terms
        self._prev_tool_pos = None
        self._prev_normal_force = 0.0

        return self._get_obs()
    def generate_targets(self):
        self.target_indices_to_ignore = []
        self.upperarm_length, self.upperarm_radius = 0.144000+0.036000, 0.036000
        self.forearm_length, self.forearm_radius = 0.108000+0.028000*2, 0.028000

        # generate capsule points
        # self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), 
        #                                                         radius=self.upperarm_radius, distance_between_points=0.03)
        # self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, -0.01]), p2=np.array([0, 0, -self.forearm_length-0.01]), 
        #                                                         radius=self.forearm_radius, distance_between_points=0.03)

        self.targets_pos_on_upperarm = self.util.capsule_points(
            p1=np.array([0, 0, 0]),
            p2=np.array([0, 0, -self.upperarm_length]),
            radius=self.upperarm_radius,
            distance_between_points=0.028
        )
        self.targets_pos_on_forearm = self.util.capsule_points(
            p1=np.array([0, 0, -0.01]),
            p2=np.array([0, 0, -self.forearm_length-0.01]),
            radius=self.forearm_radius,
            distance_between_points=0.028
        )
        
        # create target points
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.01, rgbaColor=[0, 1, 1, 1], physicsClientId=self.bc._client)
        self.targets_upperarm = []
        self.targets_forearm = []
        for _ in range(len(self.targets_pos_on_upperarm)):
            self.targets_upperarm.append(self.bc.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                           basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.bc._client))
        for _ in range(len(self.targets_pos_on_forearm)):
            self.targets_forearm.append(self.bc.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, 
                                                          basePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.bc._client))
        self.total_target_count = len(self.targets_upperarm) + len(self.targets_forearm)

        # move targets to initial positions
        self.update_targets()

        # feasible targets
        feasible_targets_found = self.get_feasible_targets_pos()

        return feasible_targets_found
    
    def remove_targets(self):
        for target in self.targets_upperarm:
            self.bc.removeBody(target, physicsClientId=self.bc._client)
        for target in self.targets_forearm:
            self.bc.removeBody(target, physicsClientId=self.bc._client)

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_shoulder, computeForwardKinematics=True, physicsClientId=self.bc._client)[4:6]
        upperarm_orient = self.util.rotate_quaternion_by_axis(upperarm_orient, axis='x', degrees=-90)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos, target_orn = self.bc.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.bc._client)
            self.targets_pos_upperarm_world.append(target_pos)
            self.bc.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.bc._client)

        forearm_pos, forearm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow, computeForwardKinematics=True, physicsClientId=self.bc._client)[4:6]
        forearm_orient = self.util.rotate_quaternion_by_axis(forearm_orient, axis='x', degrees=-90)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos, target_orn = self.bc.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.bc._client)
            self.targets_pos_forearm_world.append(target_pos)
            self.bc.resetBasePositionAndOrientation(target, target_pos, target_orn, physicsClientId=self.bc._client)

    def mark_feasible_targets(self):
        # change color for feasible targets
        for target in self.feasible_targets:
            self.bc.changeVisualShape(target, -1, rgbaColor=[0, 0.2, 1, 1], physicsClientId=self.bc._client)

    def get_feasible_targets_pos(self):
        # randomize order of trial
        self.target_order_flags = {'upperarm_front': False,
                                   'upperarm_back': False,
                                   'forearm_front': False,
                                   'forearm_back': False}
        # Runtime region override (ported from WipingEnvFixedV2): set
        # env._target_trial_order_override = ['forearm_front', ...] before reset()
        # to select region(s) without editing code. Default stays forearm_back.
        # Full set: ['upperarm_front', 'upperarm_back', 'forearm_front', 'forearm_back']
        self.target_trial_order = list(getattr(self, '_target_trial_order_override', ['forearm_back']))
        self.target_trial_order = random.sample(self.target_trial_order, len(self.target_trial_order))
        # FIX (upperarm_back hang): upperarm_back targets are generated at axis 1
        # (it is skipped at axis 0 just below). Include axis 1 ONLY when upperarm_back
        # is actually requested, so all forearm / upperarm_front runs stay byte-identical
        # (axis [0]). Without this, upperarm_back never gets targets -> infinite reset loop.
        self.target_axis_trial_order = [0, 1] if 'upperarm_back' in self.target_trial_order else [0]
        self.target_axis_trial_order = random.sample(self.target_axis_trial_order, len(self.target_axis_trial_order))

        # flag
        feasible_targets_found = False

        # split targets to front and back
        def split_targets(targets_pos, axis):
            """Splits targets into front and back halves based on the x or y coordinate. (axis = 0 or 1)"""
            z_positions = np.array([pos[axis] for pos in targets_pos])
            mid_point = np.median(z_positions)
            front_indices = [i for i, pos in enumerate(targets_pos) if pos[axis] > mid_point]
            back_indices = [i for i, pos in enumerate(targets_pos) if pos[axis] <= mid_point]
            return front_indices, back_indices

        def split_half(targets_pos, axis=0):
            """Split targets into front and back, returning indices."""
            front_half_indices, back_half_indices = split_targets(targets_pos, axis)
            return front_half_indices, back_half_indices

        # check both x and y axis
        for axis in self.target_axis_trial_order:
            front_targets_on_upperarm_indices, back_targets_on_upperarm_indices = split_half(self.targets_pos_on_upperarm, axis)
            front_targets_on_forearm_indices, back_targets_on_forearm_indices = split_half(self.targets_pos_on_forearm, axis)
            self.target_pos_dict = {'upperarm_front': np.array(self.targets_pos_on_upperarm)[front_targets_on_upperarm_indices],
                                    'upperarm_back': np.array(self.targets_pos_on_upperarm)[back_targets_on_upperarm_indices],
                                    'forearm_front': np.array(self.targets_pos_on_forearm)[front_targets_on_forearm_indices],
                                    'forearm_back': np.array(self.targets_pos_on_forearm)[back_targets_on_forearm_indices]}
            self.target_pos_world_dict = {'upperarm_front': np.array(self.targets_pos_upperarm_world)[front_targets_on_upperarm_indices],
                                          'upperarm_back': np.array(self.targets_pos_upperarm_world)[back_targets_on_upperarm_indices],
                                          'forearm_front': np.array(self.targets_pos_forearm_world)[front_targets_on_forearm_indices],
                                          'forearm_back': np.array(self.targets_pos_forearm_world)[back_targets_on_forearm_indices]}
            self.target_dict = {'upperarm_front': np.array(self.targets_upperarm)[front_targets_on_upperarm_indices],
                                'upperarm_back': np.array(self.targets_upperarm)[back_targets_on_upperarm_indices],
                                'forearm_front': np.array(self.targets_forearm)[front_targets_on_forearm_indices],
                                'forearm_back': np.array(self.targets_forearm)[back_targets_on_forearm_indices]}
            
            for order_key in self.target_trial_order:
                if order_key=='upperarm_back' and axis==0:
                    continue    # upperarm_back is generated at axis 1, not axis 0
                if axis==1 and order_key!='upperarm_back':
                    continue    # FIX: at axis 1 process ONLY upperarm_back, so other
                                # regions are not re-processed (keeps them byte-identical)

                # reset robot
                self.robot.reset()

                # set flag & targets
                self.target_order_flags[order_key] = True
                targets_pos = self.target_pos_dict[order_key]
                targets_pos_world = self.target_pos_world_dict[order_key]
                targets = self.target_dict[order_key]

                # compute world_to_target_point
                def compute_mean_target(targets_pos):
                    targets_pos = np.array(targets_pos)
                    mean_targets_pos = np.mean(targets_pos, axis=0)
                    return tuple(mean_targets_pos)

                if 'upperarm' in order_key:
                    upperarm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_shoulder, computeForwardKinematics=True, physicsClientId=self.bc._client)[5]
                    self.world_to_target_point = [compute_mean_target(targets_pos_world), upperarm_orient]
                else:
                    forearm_orient = self.bc.getLinkState(self.humanoid._humanoid, self.right_elbow, computeForwardKinematics=True, physicsClientId=self.bc._client)[5]
                    self.world_to_target_point = [compute_mean_target(targets_pos_world), forearm_orient]
                
                if 'front' in order_key and axis==1:
                    self.world_to_target_point = [self.world_to_target_point[0],
                                                  self.util.rotate_quaternion_by_axis(self.world_to_target_point[1], axis='y', degrees=90)]
                elif 'back' in order_key and axis==0:
                    self.world_to_target_point = [self.world_to_target_point[0],
                                                  self.util.rotate_quaternion_by_axis(self.world_to_target_point[1], axis='y', degrees=180)]
                elif 'back' in order_key and axis==1:
                    self.world_to_target_point = [self.world_to_target_point[0],
                                                  self.util.rotate_quaternion_by_axis(self.world_to_target_point[1], axis='y', degrees=270)]

                # compute desired world_to_eef (initial robot config)
                world_to_eef = self.bc.multiplyTransforms(self.world_to_target_point[0], self.world_to_target_point[1],
                                                          self.target_to_eef[0], self.target_to_eef[1], physicsClientId=self.bc._client)
                
                # if self.gui:
                #     self.util.draw_frame(world_to_eef[0], world_to_eef[1])  #####

                # set robot initial joint state
                q_robot = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                               lowerLimits=self.robot.arm_lower_limits, upperLimits=self.robot.arm_upper_limits, 
                                                               jointRanges=self.robot.arm_joint_ranges, restPoses=self.robot.arm_rest_poses,
                                                               maxNumIterations=40, physicsClientId=self.bc._client)
                q_robot = [q_robot[i] for i in range(len(self.robot.arm_controllable_joints))]
                if min(q_robot) < min(self.robot.arm_lower_limits) or max(q_robot) > max(self.robot.arm_upper_limits):  # invalid joint state
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue

                for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                    self.bc.resetJointState(self.robot.id, joint_id, q_robot[i], physicsClientId=self.bc._client)
                self.bc.stepSimulation(physicsClientId=self.bc._client)

                # check if config is valid
                eef_pos = self.bc.getLinkState(self.robot.id, self.robot.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[0]
                dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
                if dist > 0.03 or self.robot_in_collision(q_robot):
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue

                #####
                # compute desired world_to_eef (check if it can get closer to the target point)
                world_to_eef = self.bc.multiplyTransforms(self.world_to_target_point[0], self.world_to_target_point[1],
                                                          self.target_closer_to_eef[0], self.target_closer_to_eef[1], physicsClientId=self.bc._client)

                # if self.gui:
                #     self.util.draw_frame(world_to_eef[0], world_to_eef[1])  #####

                # set robot initial joint state
                q_robot_closer = self.bc.calculateInverseKinematics(self.robot.id, self.robot.eef_id, world_to_eef[0], world_to_eef[1],
                                                               lowerLimits=self.robot.arm_lower_limits, upperLimits=self.robot.arm_upper_limits, 
                                                               jointRanges=self.robot.arm_joint_ranges, restPoses=self.robot.arm_rest_poses,
                                                               maxNumIterations=40, physicsClientId=self.bc._client)
                q_robot_closer = [q_robot_closer[i] for i in range(len(self.robot.arm_controllable_joints))]
                if min(q_robot_closer) < min(self.robot.arm_lower_limits) or max(q_robot_closer) > max(self.robot.arm_upper_limits):  # invalid joint state
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue

                for i, joint_id in enumerate(self.robot.arm_controllable_joints):
                    self.bc.resetJointState(self.robot.id, joint_id, q_robot_closer[i], physicsClientId=self.bc._client)
                self.bc.stepSimulation(physicsClientId=self.bc._client)

                # check if config is valid
                eef_pos = self.bc.getLinkState(self.robot.id, self.robot.eef_id, computeForwardKinematics=True, physicsClientId=self.bc._client)[0]
                dist = np.linalg.norm(np.array(world_to_eef[0]) - np.array(eef_pos))
                if dist > 0.03 or self.robot_in_collision(q_robot_closer):
                    # reset flag
                    self.target_order_flags[order_key] = False
                    continue
                #####

                # Reachable-arc redefinition (flag-gated, default OFF): drop the target band
                # whose outward surface normal is averted from the robot (e.g. the bed-facing
                # underside of a supine limb), which no reachable pose can wipe. This scopes
                # the region to its physically wipeable arc so a policy can achieve ~complete
                # coverage; full-circumference coverage is left to physical limb repositioning.
                if getattr(self, "_reachable_only", False):
                    keep = self._reachable_target_mask(targets_pos_world, targets_pos)
                    tgt_list = list(targets)
                    for t, k in zip(tgt_list, keep):
                        if not k:   # hide the occluded targets so they neither count nor clutter
                            self.bc.resetBasePositionAndOrientation(
                                t, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.bc._client)
                    targets_pos = [p for p, k in zip(list(targets_pos), keep) if k]
                    targets_pos_world = [p for p, k in zip(list(targets_pos_world), keep) if k]
                    targets = [t for t, k in zip(tgt_list, keep) if k]

                self.feasible_targets_pos = targets_pos
                self.feasible_targets_pos_world = targets_pos_world
                self.feasible_targets = targets
                self.feasible_targets_count = len(targets)
                self.init_q_R = q_robot
                self.arm_side = order_key
                feasible_targets_found = True
                #print(f'arm_side: {self.arm_side}, axis: {axis}')

                # if not hasattr(self, "_dbg_arm_side_once"):
                #     self._dbg_arm_side_once = True
                #     print(f"arm_side: {self.arm_side}, axis: {self.axis}")
                                    
                break

            if feasible_targets_found:
                break

        return feasible_targets_found

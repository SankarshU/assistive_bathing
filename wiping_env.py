import os
import numpy as np
import random
import pybullet as p
from gym import spaces

from .env import AssistiveEnv


class WipingEnv(AssistiveEnv):
    def __init__(self):
        super(WipingEnv, self).__init__()
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
        self.task_success_threshold = 0.6

                # ---- reward shaping upgrades (paper-inspired) ----
        self.reach_switch_dist = 0.10     # meters; switch reach->wipe shaping
        self.near_target_dist = 0.06      # meters; dense proximity shaping
        self.contact_band_min = 2.0       # N  (start easier than 5N)
        self.contact_band_max = 12.0      # N
        self.force_weight = 0.02          # tune 0.01-0.05
        self.near_target_weight = 0.5     # tune 0.2-1.0
        self.clear_radius = 0.035         # curriculum: start >0.025, tighten later

        # continuity (optional later)
        self.prev_contact_pos = None
        self.jump_thresh = 0.10
        self.jump_gamma = 0.3

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
        #1) Apply action (advance physics + controller)
        if getattr(self, "_warmup_steps", 0) > 0:
            action = np.zeros_like(action)
            self._warmup_steps -= 1
        self.take_step(action, gains=self.robot_gains, forces=self.robot_forces)
            # ---- RAW CONTACT DIAGNOSTIC (ADD THIS) ----
        raw = self.bc.getContactPoints(
            bodyA=self.tool,
            bodyB=self.humanoid._humanoid,
            physicsClientId=self.bc._client
        )
        if not hasattr(self, "_dbg_raw_once"):
            self._dbg_raw_once = True
            #print("RAW contactPoints(tool-humanoid) =", len(raw))
            if len(raw) > 0:
                c = raw[0]
                #print("RAW first contact: linkA=", c[3], " linkB=", c[4], " fn=", float(c[9]) if len(c) > 9 else None)

        
        
        if getattr(self, "_dbg_rawc", 0) % 200 == 0:
            print("RAW contactPoints(tool-humanoid) =", len(raw))
        self._dbg_rawc = getattr(self, "_dbg_rawc", 0) + 1
        # -------------------------------------------

    
        # ---- DEBUG BLOCK HERE (leave as-is) ----
        if not hasattr(self, "_dbg_tool_links"):
            self._dbg_tool_links = True
            n = self.bc.getNumJoints(self.tool, physicsClientId=self.bc._client)
            #print("TOOL joints:", n)
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
        # ---------------------------------------
    
        # 2) Compute wiping signals from contacts (updated function)
        clears, contact_pos, normal_force, near_target_contact = self.get_new_contact_points()
        if not hasattr(self, "_dbg_contactcount"):
            self._dbg_contactcount = 0
        self._dbg_contactcount += 1
        if self._dbg_contactcount % 2000 == 0:
            cps = self.bc.getContactPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid,
                                           physicsClientId=self.bc._client)
            print("CONTACTPOINTS count(tool-human) =", len(cps))
        
        # 3) Observation after the step
        obs = self._get_obs()
    
        # 4) Distance-to-arm (phase-1): robust min over ALL humanoid links
        # hum_n = self.bc.getNumJoints(self.humanoid._humanoid, physicsClientId=self.bc._client)
        
        # best = None
        # for lb in range(-1, hum_n):  # include base (-1) and all links
        #     cps = self.bc.getClosestPoints(
        #         bodyA=self.tool,
        #         bodyB=self.humanoid._humanoid,
        #         distance=2.0,      # BIG so it doesn't go empty
        #         linkIndexA=0,      # tool link 0
        #         linkIndexB=lb,
        #         physicsClientId=self.bc._client
        #     )
        #     if cps:
        #         d = min(c[8] for c in cps)
        #         best = d if (best is None or d < best) else best
        
        # if best is None:
        #     d_arm = 2.0
        #     reward_reach = -2.0
        # else:
        #     d_arm = float(best)
        #     reward_reach = -d_arm

                # robust d_arm: min distance from tool link 0 to ANY right-arm link
        best = None
        for lb in self.human_right_arm:  # [3,4,5,6,7,8]
            cps = self.bc.getClosestPoints(
                bodyA=self.tool, bodyB=self.humanoid._humanoid,
                distance=2.0,
                linkIndexA=0,
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
                # 5) Distance-to-targets (phase-2): once close to arm, shape toward remaining markers
        if hasattr(self, "feasible_targets_pos_world") and len(self.feasible_targets_pos_world) > 0:
            tool_pos = np.array(self.bc.getLinkState(
                self.tool, 0, computeForwardKinematics=True, physicsClientId=self.bc._client
            )[0])
            d_targets_mean = float(np.mean([
                np.linalg.norm(tool_pos - np.array(tw)) for tw in self.feasible_targets_pos_world
            ]))
            reward_wipe_dist = -d_targets_mean
        else:
            reward_wipe_dist = 0.0
    
        # Two-phase switch (reach -> wipe)
        reward_distance = reward_reach if d_arm > self.reach_switch_dist else reward_wipe_dist
    
        # 6) Action penalty
        reward_action = -float(np.sum(np.square(action)))
    
        # 7) Wipe progress reward (sparse clears)
        reward_clears = float(clears)
    
        # 8) Dense proxy reward: any contact near remaining targets
        reward_near_target = float(near_target_contact)
    
        # 9) Force-band shaping (encourage safe sustained contact)
        if normal_force <= 1e-6:
            reward_force = -1.0
        else:
            if normal_force < self.contact_band_min:
                reward_force = -(self.contact_band_min - normal_force)
            elif normal_force > self.contact_band_max:
                reward_force = -(normal_force - self.contact_band_max)
            else:
                reward_force = 0.0
    
        # 10) Total reward
        reward = (
            self.distance_weight * reward_distance +
            self.action_weight * reward_action +
            self.wiping_reward_weight * reward_clears +
            self.near_target_weight * reward_near_target +
            self.force_weight * reward_force
        )
    
        # 11) Info + termination
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
        }
    
        done = bool(self.task_success >= (self.feasible_targets_count * self.task_success_threshold))
    
        # fixed debug counter
        self._dbg_t = getattr(self, "_dbg_t", 0) + 1
        if self._dbg_t % 50 == 0:
            n_t = len(self.feasible_targets_pos_world) if hasattr(self, "feasible_targets_pos_world") else 0
            print(f"d_arm={d_arm:.3f}  force={normal_force:.2f}  near={near_target_contact}  clears={clears}  n_targets={n_t}")
            print(f"d_arm={d_arm:.3f}  force={normal_force:.4f}  near={near_target_contact}  clears={clears}  n_targets={len(self.feasible_targets_pos_world)}")
    
        return obs, reward, done, info     
    def get_new_contact_points(self):
        """
        Returns:
          clears: int (#targets cleared this step)
          contact_pos: np.array or None (one representative contact position on human)
          normal_force: float (sum of normal forces for tool-human contacts this step)
          near_target_contact: int (1 if contact is within near_target_dist of any remaining target)
        """
        clears = 0
        contact_pos = None
        normal_force = 0.0
        near_target_contact = 0

        # Cache for speed
        targets_world = self.feasible_targets_pos_world

        for c in self.bc.getContactPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid, physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
            posB = np.array(c[6])          # contact position on human
            fn = float(c[9]) if len(c) > 9 else 0.0  # normal force (Bullet contact tuple)

            if linkA != 0:
                continue
            if linkB < 0 or linkB not in self.human_right_arm:
                continue

            # Accumulate force + store a representative contact point
            normal_force += fn
            contact_pos = posB

            # Dense proxy: if we are contacting near ANY remaining target -> give signal
            if len(targets_world) > 0:
                # nearest target distance
                dmin = np.min([np.linalg.norm(posB - np.array(tw)) for tw in targets_world])
                if dmin < self.near_target_dist:
                    near_target_contact = 1

            # Sparse: clear targets if within radius
            indices_to_delete = []
            for i, (target_pos_world, target) in enumerate(zip(self.feasible_targets_pos_world, self.feasible_targets)):
                if np.linalg.norm(posB - np.array(target_pos_world)) < self.clear_radius:
                    clears += 1
                    self.task_success += 1
                    self.bc.resetBasePositionAndOrientation(target, [1000, 1000, 1000], [0, 0, 0, 1], physicsClientId=self.bc._client)
                    indices_to_delete.append(i)

            if indices_to_delete:
                self.feasible_targets_pos = [t for i, t in enumerate(self.feasible_targets_pos) if i not in indices_to_delete]
                self.feasible_targets = [t for i, t in enumerate(self.feasible_targets) if i not in indices_to_delete]
                self.feasible_targets_pos_world = [t for i, t in enumerate(self.feasible_targets_pos_world) if i not in indices_to_delete]

        return clears, contact_pos, normal_force, near_target_contact

    def get_new_contact_points(self):
        """
        Returns:
          clears: int (#unique targets cleared this step)
          contact_pos: np.array or None (representative contact position on human)
          normal_force: float (sum of normal forces for tool-human contacts this step)
          near_target_contact: int (1 if ANY contact is within near_target_dist of ANY remaining target)
        """
        clears = 0
        contact_pos = None
        normal_force = 0.0
        near_target_contact = 0
    
        # If no targets remain, early exit but still compute force/contact_pos if in contact
        if (not hasattr(self, "feasible_targets_pos_world")) or (len(self.feasible_targets_pos_world) == 0):
            for c in self.bc.getContactPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid,
                                              physicsClientId=self.bc._client):
                linkA = c[3]
                linkB = c[4]
    
                # ✅ accept tool link 0 OR cloth link 1
                if linkA not in (0, 1):
                    continue
                if linkB < 0 or linkB not in self.human_right_arm:
                    continue
    
                normal_force += float(c[9]) if len(c) > 9 else 0.0
                contact_pos = np.array(c[6])
            return 0, contact_pos, float(normal_force), 0
    
        targets_world = np.asarray(self.feasible_targets_pos_world, dtype=np.float32)  # (N,3)
        to_clear = set()
    
        for c in self.bc.getContactPoints(bodyA=self.tool, bodyB=self.humanoid._humanoid,
                                          physicsClientId=self.bc._client):
            linkA = c[3]
            linkB = c[4]
    
            # ✅ accept tool link 0 OR cloth link 1
            if linkA not in (0, 1):
                continue
            if linkB < 0 or linkB not in self.human_right_arm:
                continue
    
            posB = np.asarray(c[6], dtype=np.float32)  # contact position on human
            fn = float(c[9]) if len(c) > 9 else 0.0
    
            normal_force += fn
            contact_pos = posB
    
            dists = np.linalg.norm(targets_world - posB[None, :], axis=1)
    
            if np.min(dists) < self.near_target_dist:
                near_target_contact = 1
    
            hit_idx = np.where(dists < self.clear_radius)[0]
            for i in hit_idx.tolist():
                to_clear.add(int(i))
    
        if to_clear:
            # IMPORTANT: delete from highest index downward so list indices stay valid
            for i in sorted(to_clear, reverse=True):
                target_body = self.feasible_targets[i]
                self.bc.resetBasePositionAndOrientation(
                    target_body, [1000, 1000, 1000], [0, 0, 0, 1],
                    physicsClientId=self.bc._client
                )
    
                del self.feasible_targets[i]
                del self.feasible_targets_pos_world[i]
                if hasattr(self, "feasible_targets_pos") and self.feasible_targets_pos is not None:
                    del self.feasible_targets_pos[i]
    
            clears = len(to_clear)
            self.task_success += clears
    
        return clears, (None if contact_pos is None else np.array(contact_pos)), float(normal_force), int(near_target_contact)


    def get_new_contact_points(self):
        """
        Returns:
          clears: int (#unique targets cleared this step)
          contact_pos: np.array or None (representative contact position on human)
          normal_force: float (sum of normal forces for tool-human contacts this step)
          near_target_contact: int (1 if ANY contact is within near_target_dist of ANY remaining target)
    
        Notes:
          - Accepts tool link 0 ("tool") OR link 1 ("cloth") based on your debug (linkA=0 observed).
          - Handles feasible_targets / feasible_targets_pos_world whether they are Python lists OR numpy arrays.
          - Optional: set self.min_clear_force = 0.2 (or similar) to require real contact force for clears.
        """
        clears = 0
        contact_pos = None
        normal_force = 0.0
        near_target_contact = 0
    
        # --- fetch contacts once ---
        cps = self.bc.getContactPoints(
            bodyA=self.tool,
            bodyB=self.humanoid._humanoid,
            physicsClientId=self.bc._client
        )
    
        # one-time raw debug (optional; remove later)
        if not hasattr(self, "_dbg_contact_once"):
            self._dbg_contact_once = True
            print("RAW cps:", len(cps))
            if len(cps) > 0:
                c0 = cps[0]
                print("RAW first:", "linkA=", c0[3], "linkB=", c0[4], "fn=", float(c0[9]))
    
        # Helper: list-vs-numpy delete
        def _delete_indices(obj, idxs):
            if obj is None:
                return None
            if isinstance(obj, list):
                for i in idxs:
                    del obj[i]
                return obj
            # numpy array or array-like
            try:
                return np.delete(obj, idxs, axis=0)
            except Exception:
                # fallback: try without axis
                return np.delete(obj, idxs)
    
        # Helper: length for list or numpy
        def _len(obj):
            if obj is None:
                return 0
            return len(obj)
    
        # Early exit: no targets left (still compute contact/force)
        if (not hasattr(self, "feasible_targets_pos_world")) or (_len(self.feasible_targets_pos_world) == 0):
            for c in cps:
                linkA = c[3]
                linkB = c[4]
    
                # ✅ accept tool link 0 OR cloth link 1
                if linkA not in (0, 1):
                    continue
                if linkB < 0 or linkB not in self.human_right_arm:
                    continue
    
                fn = float(c[9]) if len(c) > 9 else 0.0
                normal_force += fn
                contact_pos = np.asarray(c[6], dtype=np.float32)
    
            return 0, (None if contact_pos is None else np.array(contact_pos)), float(normal_force), 0
    
        # Targets as numpy for vectorized distance checks
        targets_world = np.asarray(self.feasible_targets_pos_world, dtype=np.float32)  # (N,3)
        to_clear = set()
    
        # Optional force gate for "real" clears (recommend keeping but start with 0.0)
        min_clear_force = float(getattr(self, "min_clear_force", 0.0))
    
        for c in cps:
            linkA = c[3]
            linkB = c[4]
    
            # ✅ accept tool link 0 OR cloth link 1
            if linkA not in (0, 1):
                continue
            if linkB < 0 or linkB not in self.human_right_arm:
                continue
    
            posB = np.asarray(c[6], dtype=np.float32)  # contact position on human
            fn = float(c[9]) if len(c) > 9 else 0.0
    
            normal_force += fn
            contact_pos = posB
    
            # distances from this contact point to all remaining targets
            dists = np.linalg.norm(targets_world - posB[None, :], axis=1)
    
            if np.min(dists) < self.near_target_dist:
                near_target_contact = 1
    
            # Only count "clears" if force is meaningful (optional but recommended)
            if fn >= min_clear_force:
                hit_idx = np.where(dists < self.clear_radius)[0]
                for i in hit_idx.tolist():
                    to_clear.add(int(i))
    
        # Apply clears
        if to_clear:
            idxs = sorted(to_clear, reverse=True)
    
            # Move target bodies away (safe whether list or numpy)
            for i in idxs:
                target_body = self.feasible_targets[i]
                self.bc.resetBasePositionAndOrientation(
                    target_body, [1000, 1000, 1000], [0, 0, 0, 1],
                    physicsClientId=self.bc._client
                )
    
            # Delete consistently
            self.feasible_targets = _delete_indices(self.feasible_targets, idxs)
            self.feasible_targets_pos_world = _delete_indices(self.feasible_targets_pos_world, idxs)
            if hasattr(self, "feasible_targets_pos") and self.feasible_targets_pos is not None:
                self.feasible_targets_pos = _delete_indices(self.feasible_targets_pos, idxs)
    
            clears = len(idxs)
            self.task_success += clears
    
        return (
            clears,
            (None if contact_pos is None else np.array(contact_pos)),
            float(normal_force),
            int(near_target_contact),
        )
    
    def _get_obs(self):
        obs = np.ones(self.obs_robot_len)*30  # padding irrelevant obs values to 30
        real_obs_len = 3 + 4 + 6 + len(self.feasible_targets_pos_world)*3

        tool_pos, tool_orn = self.bc.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.bc._client)[:2]
        tool_pos = np.array(tool_pos)
        tool_orn = np.array(tool_orn)

        robot_joint_states = self.bc.getJointStates(self.robot.id, jointIndices=self.robot.arm_controllable_joints, physicsClientId=self.bc._client)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        real_obs = np.concatenate([tool_pos, tool_orn, robot_joint_positions] + list(self.feasible_targets_pos_world)).ravel().astype(np.float32)
        obs[:real_obs_len] = real_obs

        return obs

    def reset(self):
        self._warmup_steps = 30
        self.bc.resetSimulation(self.bc._client)
        self.setup_timing()
        self.create_world()
    
        self.task_success = 0
        self.contact_points_on_arm = {}
        self.robot_lower_limits = self.robot.arm_lower_limits
        self.robot_upper_limits = self.robot.arm_upper_limits
    
        feasible_targets_found = False
        while True:
            # randomize human arm config and check for collisions
            self.robot.reset()
            self.reset_and_check()
    
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
    
        return self._get_obs()        
    def generate_targets(self):
        self.target_indices_to_ignore = []
        self.upperarm_length, self.upperarm_radius = 0.144000+0.036000, 0.036000
        self.forearm_length, self.forearm_radius = 0.108000+0.028000*2, 0.028000

        # generate capsule points
        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), 
                                                                radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, -0.01]), p2=np.array([0, 0, -self.forearm_length-0.01]), 
                                                                radius=self.forearm_radius, distance_between_points=0.03)
        
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
        #self.target_trial_order = ['upperarm_front', 'upperarm_back', 'forearm_front', 'forearm_back']
        self.target_trial_order = [ 'forearm_back']
        self.target_trial_order = random.sample(self.target_trial_order, len(self.target_trial_order))
        #self.target_axis_trial_order = [0, 1]
        self.target_axis_trial_order = [0]
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
                    continue    

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

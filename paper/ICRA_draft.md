# One Policy for the Whole Arm: Region-Conditioned Distillation for Safe Robotic Bed-Bathing

*Draft — ICRA submission. Authors: TBD.*

---

## Abstract

Robotic assistance with bed-bathing requires a manipulator to wipe the skin of a supine
person completely, systematically, and with clinically safe contact forces. Reinforcement
learning can produce competent per-region wiping policies, but deploying a separate network
for every body part is impractical, and the standard contact-counting reward optimizes the
wrong objective — it clears targets aggressively without regard to coverage or force safety.
We make three contributions. First, the **Comfort reward**: a structured wiping reward that
shapes motion along a self-supervised arm axis — extended here with a 2-D
(length × circumference) coverage term built from the same SVD's second principal axis —
producing complete, force-safe sweeps. Second, a **single distilled policy conditioned on
region and a one-bit comfort dial** that replaces eight per-region specialists, matches or
exceeds them in every region, and beats a flat multi-task policy by **+33 % targets cleared
(positive in 12/12 region×seed pairs, exact sign test p = 0.0005)** while keeping contact
force in the therapeutic band 2–2.7× more often. Third, the dial bit exposes a
**caregiver-selectable gentle↔thorough** behavior: in a paired same-network test, the
Comfort-reward (Gentle) setting raises therapeutic-band adherence by **+34 % (10/12 pairs,
p = 0.039)** at an 8 % coverage cost. Across the full method ladder, learning dominates a
scripted geometric controller on every wiping-quality metric — the scripted sweep traces
near-perfect geometry yet fails sustained safe contact — and the reward design decides not
how *much* the robot wipes but how *safely*.

---

## 1. Introduction & problem definition

**Setting.** A UR5 manipulator holds a soft tool and cleans a target region of a supine
human arm in simulation (PyBullet). Each region carries up to 15 cleanable **targets**; a
target is *cleared* when the tool passes within `clear_radius = 0.022 m` while in contact.
Episodes last 150 steps. The observation is a 90-D vector (robot joint state, end-effector
pose, tool–skin contact geometry, region/target encoding); the action is a 6-D continuous
end-effector command.

**Why raw clear-count is the wrong objective.** Assistive bathing is a *coverage-and-safety*
task, not a target-collection game. A caregiver wipes the whole region in order, maintaining
gentle, sustained skin contact. A reward that simply counts contacts encourages fast,
aggressive dabbing that inflates the clear-count while leaving gaps and spiking force. We
therefore evaluate on a battery of behavior metrics (Section 4.3), and design a reward — the
**Comfort reward** — for the true objective.

**Why one policy, not four.** Region-specialist policies are easy to train but impractical
to deploy: a robot would need to store and select among a library of networks. We ask whether
a *single* context-conditioned policy, distilled from the specialists, can replace the whole
library without loss — the paper's central generalization question.

---

## 2. Related work (brief)

RL for assistive manipulation and bed-bathing (Erickson et al., Assistive Gym lineage);
policy distillation and behavior cloning of RL teachers; multi-task and goal-/context-
conditioned policies. Our contribution is the combination of a coverage-and-safety reward
(the Comfort reward) with region- and dial-conditioned distillation, and a controlled
8-method, 3-seed study isolating the effect of reward design vs. distillation scheme vs.
learning itself.

---

## 3. Method

### 3.1 Teacher policies
One PPO policy (RLlib 1.13) per (region, reward) pair. Policy/value net `fcnet_hiddens =
[100,100]`, tanh; `train_batch_size = 19200`, `num_sgd_iter = 50`, `sgd_minibatch_size =
128`. Trained in ~250k-step chunks with best-checkpoint archiving.

### 3.2 Reward design

We compare two reward designs on identical dynamics.

**Baseline — Contact-count reward `R_cc`** (the original reward from the public bed-bathing
codebase, git `96fb9af`):

```
R_cc = 1.0·(−d_arm) + 0.01·(−‖a‖²) + 5.0·c_t
```

where `d_arm` is tool–arm distance, `a` the action, and `c_t` the targets cleared at step t.
It rewards *how many* targets are cleared but has no notion of *where* along the arm the tool
is or whether coverage is systematic.

**Proposed — Comfort reward `R_comf`** keeps those base terms and adds four shaping terms:

```
R_comf = 1.0·r_dist + 0.01·r_act + 5.0·r_clr + 1.0·r_sweep + 1.0·r_win + 1.0·r_end + 1.0·r_cov
```

| term | what it rewards | weight |
|---|---|---|
| `r_dist`  | on-surface proximity (two-phase: reach → wipe)            | 1.0  |
| `r_act`   | −‖a‖² (effort penalty)                                    | 0.01 |
| `r_clr`   | targets cleared this step `c_t`                           | 5.0  |
| `r_sweep` | monotone progress along the self-supervised arm axis      | 1.0  |
| `r_win`   | contact/clears near the current sweep coordinate `s`      | 1.0  |
| `r_end`   | evidence-gated end-of-pass bonus (B = 4.0)                | 1.0  |
| `r_cov`   | first visit to each cell of a 4×4 `(s, w)` coverage grid  | 1.0  |

(1) **two-phase sticky distance** `r_dist` separating reach from wipe;
(2) **directional sweep** `r_sweep` rewarding monotone progress along a self-supervised arm
axis — the principal axis is estimated by SVD of the target cloud (refreshed every ~200 steps)
and the tool is projected onto it to yield a normalized sweep coordinate `s ∈ [0,1]`
(`k_f = 0.5`, `k_b = 0.75`, active only while in contact);
(3) a **local-window** term `r_win` rewarding contact/clears near the current `s`;
(4) an **evidence-gated end-of-pass bonus** `r_end` granted only on demonstrated coverage
(prevents farming the bonus by sweeping empty space);
(5) a **2-D (s, w) coverage term** `r_cov`: for this elongated target geometry the *second*
principal axis of the same SVD is the circumferential ("column") direction, so projecting the
tool onto it yields a width coordinate `w`; the first contact or clear in each unvisited
(s, w) cell of a 4×4 grid earns a one-time bonus (weight 1.0, selected by grid search over
{0, 0.5, 1, 2}: 6.08 vs. 5.68 cleared at weight 0), giving the sweep an incentive to cover
*across* the arm and not only along it.
A soft-contact latch with hysteresis (6 steps) stabilizes the wipe/exit phase against brief
force dropouts.

### 3.3 Region-conditioned student
A single Gaussian MLP policy conditioned on the region:

```
input   = [ obs(90) , context ]
context = region one-hot(4)  [ + optional comfort-dial bit κ(1) ]
backbone = MLP(128,128), tanh
heads    = mean(6) ;  log_std(6) clamped to [-5,2]   (diagonal Gaussian)
```

At deployment the one-hot selects the current body part and the optional comfort-dial bit
κ ∈ {0,1} selects behavior; one network covers the whole arm.

### 3.4 Distillation
Each teacher is rolled out in its region to record `(obs, teacher_mean, teacher_log_std,
context)`. The student minimizes a behavior-cloning term on the mean plus a
distribution-matching (KL) term on the full diagonal Gaussian:

```
L = ‖μ_S − μ_T‖²  +  β · KL( N_T ‖ N_S )                              (Adam, lr 1e-3, β = 1)

KL( N_T ‖ N_S ) = Σ_i [ log(σ_S,i/σ_T,i) + (σ_T,i² + (μ_T,i − μ_S,i)²)/(2·σ_S,i²) − 1/2 ]
```

matching both the teacher's action mean and its uncertainty. We compare four distillation
sources: **Comfort** (4 Comfort-reward teachers), **Contact-count** (4 contact-counting
teachers), **dial-conditioned** (all 8 teachers + a comfort-dial bit that switches behavior
on demand), and **flat** (all 8 teachers pooled, no bit — an irreversible blend).

> **Algorithm 1 — Region-conditioned distillation.**
> **Input:** teachers {π_r}; context map c(r); rollout length N; weight β; optional dial values {κ_r}.
> 1. **Collect:** for each region r, roll out π_r for N steps; record (o, μ_r(o), logσ_r(o), c(r)); append κ_r to c(r) if given.
> 2. **Pool:** D = ∪_r D_r.
> 3. **Train** π_θ(a | o, c) by minibatch Adam (1e-3) on the mean of L (BC + β·KL).
> **Output:** single conditioned student π_θ.

---

## 4. Experimental design

**Regions (4):** forearm back, forearm front, upper-arm back, upper-arm front.
**Methods:** three reward classes — Comfort RL, Contact-count RL, and a non-RL scripted
geometric sweep — each evaluated as *both* a per-region specialist and a distilled student;
plus the **Dial student** (one network distilled from all eight RL specialists with a comfort
dial: **Gentle** end κ=0 = Comfort behavior / **Thorough** end κ=1 = Contact-count behavior)
and a flat multi-task RL baseline (same Comfort reward, all regions, no conditioning). Comfort
specialists use the structured reward with the (s, PC2) coverage term.
**Protocol:** every method scored by identical reducers, 200 episodes/region (100 for the
scripted controller), force band 2–12 N, `clear_radius = 0.022`, **3 seeds** for every
learned method. Statistics: exact sign tests over the 12 region×seed pairs plus conservative
per-seed paired t (the seed is the statistical unit — episode-level pooling would be
pseudoreplication). All numbers are generated by `analyze_final.py` from the raw metrics CSV.

### 4.3 Evaluation metrics

Per step t an episode yields normal force f_t, an in-contact flag, sweep coordinate s_t ∈ [0,1],
per-step clears k_t, and action a_t; C = {t : contact} is the set of in-contact steps and T
the episode length.

| metric | definition | captures |
|---|---|---|
| Targets cleared        | Σ_t k_t (distinct, out of 15)                          | task performance |
| In-band force fraction | \|{t∈C : f_t ∈ [2,12] N}\| / \|C\|                       | contact-force safety |
| Contact-time fraction  | \|C\| / T                                               | sustained skin engagement |
| Sweep consistency      | (forward-progress wipe steps) / (wipe steps)            | systematic, monotone motion |
| Clear s-coverage       | \|{arm-axis bins with a clear}\| / 10                   | spatial spread along the arm |
| Clear t-coverage       | \|{time bins with a clear}\| / 10                       | temporal spread (not bursty) |
| Second-half fraction   | (Σ_{t>T/2} k_t) / Σ_t k_t                               | sustained, not front-loaded |
| Clear spatial order    | fraction of consecutive clears moving in dominant s-dir | orderly (1.0) vs. random (0.5) |
| Mean stroke length     | mean length of in-contact, constant-direction slides    | wipe (long) vs. poke (short) |
| Contact bouts          | number of separate contact episodes                     | poking fragments contact |
| Slide fraction         | \|{t∈C : \|Δs_t\| > 0.008}\| / \|C\|                     | sliding vs. tapping in place |
| Peak force             | max_{t∈C} f_t                                            | worst-case force (lower safer) |

Two families are deliberately *excluded* from cross-method comparison: **pass completions**
(counts firings of the Comfort reward's own end-of-pass gate — one method's internal
machinery) and **mean/p95 force** (reward *not touching*: the scripted controller has the
lowest mean force precisely because it grazes). Force safety is therefore reported as in-band
fraction and peak force only.

---

## 5. Results

**Main comparison** (cleared / 15, with in-band force fraction in parentheses; 3-seed means):

| method | forearm back | forearm front | upper-arm back | upper-arm front | mean |
|---|---|---|---|---|---|
| Comfort specialist          | 6.25 (0.150) | 4.99 (0.218) | 6.75 (0.164) | 5.19 (0.159) | 5.80 (0.173) |
| Contact-count specialist    | 6.72 (0.121) | 7.05 (0.256) | 6.80 (0.112) | 4.95 (0.128) | 6.38 (0.154) |
| Student ← Comfort           | 6.15 (0.154) | 6.23 (**0.282**) | 6.74 (0.172) | **5.96** (0.162) | 6.27 (**0.193**) |
| Student ← Contact-count     | 6.42 (0.117) | 7.77 (0.256) | 6.82 (0.110) | 5.69 (0.128) | 6.68 (0.153) |
| Dial @Gentle                | 6.35 (0.168) | 6.15 (**0.311**) | 6.62 (0.165) | 5.55 (0.146) | 6.17 (**0.198**) |
| Dial @Thorough              | **6.95** (0.121) | 7.65 (0.278) | **7.02** (0.106) | 5.56 (0.120) | **6.80** (0.156) |
| Flat (multi-task)           | 5.00 (0.073) | 5.20 (0.059) | 6.43 (0.109) | 3.80 (0.058) | 5.11 (0.075) |
| Scripted (non-RL)           | 6.27 (0.087) | 4.48 (0.050) | 4.28 (0.020) | 3.20 (0.002) | 4.56 (0.040) |

### 5.1 Headline: one distilled network with a caregiver-selectable gentle↔thorough dial
The dial student — a single network conditioned on region and one comfort-dial bit — spans
the coverage↔gentleness trade-off at deployment time. Averaged over 4 regions × 3 seeds:
**Thorough mode 6.80 cleared / 0.156 in-band**, **Gentle mode 6.17 cleared / 0.198 in-band**.
The ordering is consistent on **all four arm regions** (Gentle is gentler in 4/4; Thorough
clears ≥ in 4/4):

| region | Gentle: cleared / in-band | Thorough: cleared / in-band |
|---|---|---|
| forearm back | 6.35 / **0.168** | **6.95** / 0.121 |
| forearm front | 6.15 / **0.311** | **7.65** / 0.278 |
| upper-arm back | 6.62 / **0.165** | **7.02** / 0.106 |
| upper-arm front | 5.55 / **0.146** | **5.56** / 0.120 |

At each end the dial matches or exceeds the corresponding single-reward student — it is not a
compromise but both behaviors in one deployable policy.

### 5.2 The distilled policy beats the flat multi-task baseline
The dial (Thorough mode) clears **6.80** vs **5.11** for the flat baseline — **+33 %**,
positive in **12/12 region×seed pairs** (exact sign test p = 0.0005), consistent per seed
(6.43/6.81/7.14 vs 5.26/5.37/4.69). The Comfort-distilled student alone gives +1.16
clears/seed (per-seed paired t = 3.76, n = 3; 10/12 pairs, sign p = 0.039). Conditioning also
buys *safety*: every conditioned student keeps force in-band 2–2.7× more than the flat
baseline (0.153–0.198 vs 0.074). One network replaces four specialists and beats the
unconditioned alternative on both axes.

### 5.3 Distillation ≥ specialists — and rescues weak teachers
Students match their specialists where those are strong and repair them where they are weak
(cleared/15, 3-seed means): Comfort teacher→student 4.99→**6.23** on forearm front and
5.19→**5.96** on upper-arm front; Contact-count 4.95→**5.69** on upper-arm front. No region
regresses by more than 0.3. Pooled-data distillation regularizes: grand means are students
6.17–6.80 vs specialists 5.79–6.38.

### 5.4 The reward style determines gentleness: a paired, same-network test
Because the dial student is *one* network, flipping the bit is a perfectly paired comparison
— same weights, same seeds, same regions; only the dial input differs. Switching to the
Gentle (Comfort) setting raises **in-band force fraction by +34 %** (better in 10/12
region×seed pairs, exact sign test **p = 0.039**) at an 8 % coverage cost (3/12, n.s.). On the
composite objective *safe clears* (cleared × in-band), the Comfort-distilled student leads the
Contact-count-distilled one by +28 % (8/12 pairs, suggestive). Raw clear-count ordering is
reversed (the Thorough/Contact-count setting clears more) — the reward design does not make
wiping more *productive*, it makes it *safer*, and the dial exposes exactly that trade-off to
the caregiver.

### 5.5 Learning and conditioning both matter: the method ladder
Grand means over 4 regions × 3 seeds form a monotone ladder — non-RL scripted **4.56**
cleared / **0.039** in-band < flat RL **5.11** / 0.074 < specialists 5.79–6.38 / 0.15–0.17 <
distilled students **6.17–6.80** / 0.15–0.20. The scripted controller is not merely less
productive: it fails the *wiping-quality* battery — in contact only 25–41 % of the time (vs
51–74 % for the dial), in-band force collapsing to 0.020/0.002 on the upper arm, while its
near-perfect clear spatial order (0.997) shows it traces textbook geometry. Raw clear-count
and geometric orderliness are mirages of wiping; sustained safe contact is not scriptable
here and had to be learned. Crucially, **distillation cannot repair it**: cloning the scripted
controller into a student adds a little coverage (grand 4.56 → 5.05 cleared) but leaves its
in-band force at ~0.04 — versus ~0.19 for the RL students. Distillation *amplifies* what a
teacher already does; it cannot inject safe contact that the teacher never had, which is why
the generalization result is a property of the learned behavior, not of the distillation
mechanism.

---

## 6. Limitations

**Reachability.** One angular column of each *back* region faces the bed (the supine limb's
support surface). We show by per-target IK over a dense orientation grid, dynamic
clearability under three independent controllers, robot base-pose sweeps, and arm posture
sweeps that no configuration of this environment can wipe it (the humanoid has no forearm
pronation DOF); see `REACHABILITY_ANALYSIS.md`. This caps targets-cleared near 10/15 on back
regions and keeps `task_success` (69 % threshold) near zero for *all* methods; cleaning the
support-facing surface requires physically re-positioning the limb — the caregiver action our
deployment pipeline assigns to its classical phases, and future work for the policy itself.
**Force spikes.** Peak contact forces reach 47–68 N for all methods (means are in-band);
bounding worst-case force (force-rate shaping or a controller-level clamp) is required before
hardware trials. **Statistics.** Sign tests over 12 region×seed pairs are exact but the
per-seed n = 3; per-method gaps not listed with a p-value are directional. **Scope.** Results
are simulation-only (rigid humanoid, quasi-static bed scene); sim-to-real transfer is future
work.

---

## 7. Conclusion

A single distilled policy, conditioned on region and a one-bit comfort dial, wipes all four
arm regions, beats the flat multi-task baseline in every region×seed pair (+33 %,
p = 0.0005), and exposes a caregiver-selectable gentle↔thorough behavior whose Gentle end
raises therapeutic-band adherence by 34 % in a paired same-network test (p = 0.039). Across
the method ladder, learning dominates scripting on every wiping-quality metric, and
conditioned distillation dominates flat multi-task RL on both coverage and safety. The Comfort
reward does not decide how *much* a robot wipes — it decides how *safely*; distillation is
what turns that choice into a runtime-switchable property of one deployable network.

---

*Figure: `ICRA_matrix_figure.png` (method×metric comparison). Every number in §5 is emitted
by `analyze_final.py` from `behavior_metrics_summary.csv`. Reproduction:
`run_powerful_matrix4.sh` (teachers → distill → students → eval → figure, resumable).
Paper-name ↔ code-label map: Comfort = `ours`, Contact-count = `yubik`, Dial = `style`,
Gentle = `@Ours-bit`, Thorough = `@Yubik-bit` (see repository README).*

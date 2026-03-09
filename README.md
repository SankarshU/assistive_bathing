### Train from an existing checkpoint

Resume PPO training from the saved policy directory:

```bash
python learn.py --train --algo ppo \
  --load-policy-path ./trained_models/ppo/learn_bathing:WipingEnv-v0/ \
  --train-timesteps 450000
```

Evaluate the current policy:

```bash
python learn.py --evaluate
```

Render the current policy:

```bash
python learn.py --render
```

---

## Reward design

The reward combines reaching, wiping, contact quality, and directional wiping structure.

### Total reward

```text
r_t =
    distance_weight * r_dist
  + action_weight * r_act
  + wiping_reward_weight * r_clear
  + near_target_weight * r_near
  + force_weight * r_force
  + r_sweep
  + r_window
```

### Distance term

Two modes are used:

```text
r_dist =
    -d_arm                                  if not in wipe mode
    -mean distance(tool, remaining targets) if in wipe mode
```

Meaning:
- before wiping, move toward the arm
- during wiping, stay near the remaining targets

### Action penalty

```text
r_act = -||action||^2
```

This discourages jerky or unnecessarily large motions.

### Clear reward

```text
r_clear = number of targets cleared this step
```

This is the main task-progress reward.

### Near-target reward

```text
r_near = 1 if valid contact is near any remaining target, else 0
```

This gives a dense signal even before a full clear happens.

### Force-band reward

Let:
- `F_t` = valid tool-arm normal force
- `F_min` = `contact_band_min`
- `F_max` = `contact_band_max`

Then:

```text
r_force =
    0                 if force is near zero and wipe mode is active
   -1                 if force is near zero and wipe mode is inactive
   -(F_min - F_t)     if 0 < F_t < F_min
    0                 if F_min <= F_t <= F_max
   -(F_t - F_max)     if F_t > F_max
```

This encourages useful but not excessive contact.

### Stage 1: arm-axis sweep reward

A normalized arm-axis coordinate `s_t in [0,1]` is computed from the active target cloud, and a persistent wipe direction `pass_dir in {+1, -1}` is maintained.

```text
ds     = s_t - s_(t-1)
ds_dir = pass_dir * ds
```

Sweep reward:

```text
r_sweep =
    +k_f * min(ds_dir, ds_cap)              if ds_dir > sweep_eps
    -g_t * k_b * min(-ds_dir, ds_cap)       if ds_dir < -sweep_eps
     0                                      otherwise
```

Meaning:
- reward forward wiping motion
- ignore tiny jitter
- penalize meaningful backward slip

### Stage 2A: directional window reward

A local active window is defined along the same arm-axis:
- forward pass: `[s_t, s_t + Delta_w]`
- reverse pass: `[s_t - Delta_w, s_t]`

Let:
- `I_win_near` = 1 if contact is near a target in the active window, else 0
- `C_win` = number of clears inside that active window

Then:

```text
r_window =
    alpha * I_win_near + beta * C_win
```

This is applied only when wipe mode is active and contact is latched.

This encourages local, sequential wiping rather than isolated point touching.

---

## Target geometry

The visible purple targets are sampled on the arm and then split into one active region per episode:

- `upperarm_front`
- `upperarm_back`
- `forearm_front`
- `forearm_back`

The policy is rewarded for clearing the currently active target set, not the whole arm at once.

---

## Intuition

In simple terms, the policy is encouraged to:

1. reach the selected arm region
2. maintain stable contact
3. stay near remaining targets
4. use reasonable force
5. clear targets
6. wipe in a visible directional pass instead of random local poking

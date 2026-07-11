#!/usr/bin/env python3
"""
distill.py — context-conditioned policy distillation across REGIONS.

Distills one or more region teachers (PPO checkpoints) into a single student
policy  pi(a | o, c),  where c = one-hot region context. Same frozen reward.

Goal (paper Claim 4+5): one student wipes forearm_back / forearm_front /
upperarm_back, matching each specialist teacher and beating a flat all-region policy.

Three modes:
  collect : roll out a teacher in its region, record (obs, teacher action mean+log_std, context)
  train   : fit the student on the pooled collected data (BC + Gaussian-distill KL loss)
  eval    : run the student in each region, report targets_cleared / task_success

collect + eval need pybullet/ray (run locally). train is pure PyTorch (no sim).

------------------------------------------------------------------------------
EASY TEST (validate the whole pipeline NOW on the one existing teacher):
  conda activate learnbath; cd learn-bathing

  # 1) collect from the archived forearm_back winner
  python distill.py collect \
    --checkpoint trained_models/ARCHIVE_full_8term_winner_6p9_2026-06-12/checkpoint_000014/checkpoint-14 \
    --region forearm_back --episodes 100 --out distill_data/forearm_back.npz

  # 2) train a (trivially 1-context) student
  python distill.py train --data "distill_data/*.npz" --out student.pt --epochs 30

  # 3) eval: student on forearm_back should reproduce ~6.8 cleared
  python distill.py eval --student student.pt --regions forearm_back --episodes 100

FULL RUN (after training forearm_front + upperarm_back teachers via auto_loop):
  collect from all 3 teachers -> 3 npz files -> train -> eval --regions forearm_back forearm_front upperarm_back
------------------------------------------------------------------------------
"""
import argparse, atexit, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_NAME = "learn_bathing:WipingEnv-v0"
OVERRIDE_FILE = os.path.join(HERE, "_env_overrides.json")

# Default context order for 2-region work (one-hot index = position). Multi-region
# runs pass --contexts explicitly. Kept at 2 so existing 2-region students stay valid.
ALL_REGIONS = ["forearm_back", "forearm_front"]
# All valid regions (env-axis fix 2026-06-29 enabled the upper-arm pair) — used only
# to validate the --region argument, NOT as the default context order.
ALL4_REGIONS = ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]

OBS_DIM = 90
ACT_DIM = 6


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def context_vec(region, regions):
    v = np.zeros(len(regions), dtype=np.float32)
    v[regions.index(region)] = 1.0
    return v


def _set_env_overrides(region):
    """Pin reward + clear_radius + region for this run; restore file on exit."""
    prev = None
    if os.path.exists(OVERRIDE_FILE):
        with open(OVERRIDE_FILE) as f:
            prev = f.read()
    ov = {"use_extended_reward": False, "clear_radius": 0.022,
          "_target_trial_order_override": [region]}
    with open(OVERRIDE_FILE, "w") as f:
        json.dump(ov, f)
    print(f"[distill] env overrides set: {ov}")

    def _restore():
        if prev is None:
            try: os.remove(OVERRIDE_FILE)
            except OSError: pass
        else:
            with open(OVERRIDE_FILE, "w") as f:
                f.write(prev)
        print("[distill] restored _env_overrides.json")
    atexit.register(_restore)


def _teacher_dist(policy, obs):
    """Return (mean[6], log_std[6]) of the teacher's DiagGaussian for one obs.
    Reads the torch model logits directly (robust across RLlib 1.x info-key changes):
    continuous PPO has num_outputs = 2*act_dim -> [means, log_stds]."""
    import torch
    model = policy.model
    with torch.no_grad():
        logits, _ = model({"obs": torch.as_tensor(obs[None], dtype=torch.float32)}, [], None)
    logits = logits.cpu().numpy()[0]
    assert logits.shape[0] == 2 * ACT_DIM, \
        f"expected {2*ACT_DIM} logits (mean+log_std), got {logits.shape[0]}"
    return logits[:ACT_DIM], logits[ACT_DIM:]


# --------------------------------------------------------------------------- #
# MODE: collect
# --------------------------------------------------------------------------- #
def collect(args):
    _set_env_overrides(args.region)
    import multiprocessing, ray
    from learn import make_env, load_policy

    if "checkpoint" not in args.checkpoint:
        sys.exit("--checkpoint must be a checkpoint FILE path (weights-only loader).")

    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    env = make_env(ENV_NAME, seed=args.seed)
    agent, _ = load_policy(env, "ppo", ENV_NAME, args.checkpoint, seed=args.seed,
                           extra_configs={"num_workers": 0, "num_gpus": 0}, render=True)
    policy = agent.get_policy()

    regions = args.contexts.split(",") if args.contexts else ALL_REGIONS
    cvec = context_vec(args.region, regions)
    # style-conditioned distillation: append a style scalar (e.g. ours=0, yubik=1) so ONE
    # student can reproduce EITHER teacher on demand (no blending). Off unless --style given.
    if getattr(args, "style", None) is not None:
        cvec = np.concatenate([cvec, [float(args.style)]]).astype(np.float32)

    obs_buf, mean_buf, logstd_buf = [], [], []
    cleared = []
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            mean, log_std = _teacher_dist(policy, np.asarray(obs, dtype=np.float32))
            obs_buf.append(np.asarray(obs, dtype=np.float32))
            mean_buf.append(mean); logstd_buf.append(log_std)
            # step with the teacher's deterministic action (mean), tanh-free: env clips
            obs, _, done, info = env.step(mean)
        cleared.append(info.get("number_of_targets_cleared", 0))
        if (ep + 1) % 10 == 0:
            print(f"[collect] {ep+1}/{args.episodes} eps, last cleared={cleared[-1]}")
        sys.stdout.flush()
    env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ctx = np.tile(cvec, (len(obs_buf), 1)).astype(np.float32)
    np.savez_compressed(args.out,
                        obs=np.array(obs_buf, dtype=np.float32),
                        mean=np.array(mean_buf, dtype=np.float32),
                        log_std=np.array(logstd_buf, dtype=np.float32),
                        context=ctx,
                        region=args.region,
                        regions=np.array(regions))
    print(f"[collect] region={args.region} steps={len(obs_buf)} "
          f"mean_cleared={np.mean(cleared):.2f} -> {args.out}")


# --------------------------------------------------------------------------- #
# student network
# --------------------------------------------------------------------------- #
def _build_student(n_ctx, hidden=(128, 128)):
    import torch.nn as nn
    layers, d = [], OBS_DIM + n_ctx
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    backbone = nn.Sequential(*layers)
    mean_head = nn.Linear(d, ACT_DIM)
    log_std_head = nn.Linear(d, ACT_DIM)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone, self.mean_head, self.log_std_head = backbone, mean_head, log_std_head
        def forward(self, x):
            z = self.backbone(x)
            return self.mean_head(z), self.log_std_head(z).clamp(-5.0, 2.0)
    return Student()


def _kl_diag_gauss(mp, lsp, mq, lsq):
    """KL( teacher p || student q ) for diagonal Gaussians, summed over action dims.
    All args torch tensors [B, ACT_DIM]; ls = log_std."""
    import torch
    vp, vq = torch.exp(2 * lsp), torch.exp(2 * lsq)
    return (lsq - lsp + (vp + (mp - mq) ** 2) / (2 * vq) - 0.5).sum(dim=-1)


# --------------------------------------------------------------------------- #
# MODE: train  (pure PyTorch, no sim)
# --------------------------------------------------------------------------- #
def train(args):
    import torch
    files = sorted(glob.glob(args.data))
    if not files:
        sys.exit(f"no data files match {args.data}")
    O, M, L, C = [], [], [], []
    regions = None
    for f in files:
        d = np.load(f, allow_pickle=True)
        O.append(d["obs"]); M.append(d["mean"]); L.append(d["log_std"]); C.append(d["context"])
        regions = list(d["regions"]) if regions is None else regions
        print(f"[train] loaded {f}: {len(d['obs'])} steps, region={d['region']}")
    O = np.concatenate(O); M = np.concatenate(M); L = np.concatenate(L); C = np.concatenate(C)
    n_ctx = C.shape[1]
    print(f"[train] pooled {len(O)} steps, n_ctx={n_ctx}, regions={regions}")

    X = torch.tensor(np.concatenate([O, C], axis=1), dtype=torch.float32)
    Tm = torch.tensor(M, dtype=torch.float32)
    Tl = torch.tensor(L, dtype=torch.float32)

    n = len(X); idx = np.random.permutation(n); n_val = int(0.1 * n)
    vi, ti = idx[:n_val], idx[n_val:]
    student = _build_student(n_ctx)
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)

    def batch_loss(sel):
        sm, sl = student(X[sel])
        bc = ((sm - Tm[sel]) ** 2).sum(dim=-1).mean()
        kl = _kl_diag_gauss(Tm[sel], Tl[sel], sm, sl).mean()
        return bc + args.beta * kl, bc.item(), kl.item()

    for ep in range(args.epochs):
        student.train()
        np.random.shuffle(ti)
        for s in range(0, len(ti), args.batch):
            sel = torch.as_tensor(ti[s:s + args.batch])
            opt.zero_grad()
            loss, _, _ = batch_loss(sel)
            loss.backward(); opt.step()
        student.eval()
        with torch.no_grad():
            vloss, vbc, vkl = batch_loss(torch.as_tensor(vi))
        print(f"[train] epoch {ep+1}/{args.epochs}  val_loss={vloss.item():.4f} "
              f"(bc={vbc:.4f} kl={vkl:.4f})")
        sys.stdout.flush()

    torch.save({"state_dict": student.state_dict(), "n_ctx": n_ctx,
                "regions": regions, "hidden": (128, 128)}, args.out)
    print(f"[train] saved student -> {args.out}")


# --------------------------------------------------------------------------- #
# MODE: eval  (student in the sim, per region)
# --------------------------------------------------------------------------- #
def eval_student(args):
    import torch, multiprocessing, ray, csv, importlib.util
    from learn import make_env
    # reuse metrics_report's reducers so student is scored on the SAME behavior
    # metrics as the RLlib teachers/flat baseline (sweep consistency, force, coverage...).
    spec = importlib.util.spec_from_file_location("mr", os.path.join(HERE, "metrics_report.py"))
    mr = importlib.util.module_from_spec(spec); spec.loader.exec_module(mr)

    ckpt = torch.load(args.student, map_location="cpu")
    regions_in_student = ckpt["regions"]
    n_ctx = ckpt["n_ctx"]
    has_style = n_ctx > len(regions_in_student)   # student was trained with a style bit
    student = _build_student(n_ctx, tuple(ckpt["hidden"]))
    student.load_state_dict(ckpt["state_dict"]); student.eval()
    if has_style:
        print(f"[eval] style-conditioned student: evaluating at style={args.style}")

    rows = []
    for region in args.regions:
        _set_env_overrides(region)
        ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
        env = make_env(ENV_NAME, seed=args.seed)
        cvec = context_vec(region, regions_in_student)
        if has_style:
            cvec = np.concatenate([cvec, [float(args.style)]]).astype(np.float32)
        per_ep = []
        for ep in range(args.episodes):
            obs = env.reset(); done = False
            steps, acts = [], []
            while not done:
                a = np.concatenate([np.asarray(obs, np.float32), cvec])
                with torch.no_grad():
                    mean, _ = student(torch.tensor(a[None], dtype=torch.float32))
                action = mean.numpy()[0]
                obs, _, done, info = env.step(action)
                steps.append(info); acts.append(action)
            per_ep.append(mr.episode_metrics(steps, acts, args.force_lo, args.force_hi))
        env.close()

        summ = mr.summarize(per_ep)
        label = f"{getattr(args, 'tag', 'student')}_{region}"
        # per-episode CSV + append to the shared summary table (same file as metrics_report)
        per_path = os.path.join(HERE, f"metrics_per_episode_{label}.csv")
        with open(per_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_ep[0].keys())); w.writeheader(); w.writerows(per_ep)
        summ_row = {"label": label, "episodes": args.episodes,
                    "force_band": f"[{args.force_lo},{args.force_hi}]",
                    "checkpoint": args.student, **summ}
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
        mr.print_markdown(label, summ)
        rows.append((region, summ["targets_cleared_mean"], summ["targets_cleared_std"],
                     summ["sweep_consistency_mean"], summ["peak_contact_force_mean"]))

    print("\n### Student summary (cleared | sweep | peak force)\n")
    print("| region | cleared | sweep consistency | peak force |\n|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.2f} ± {r[2]:.2f} | {r[3]:.3f} | {r[4]:.1f} |")


# --------------------------------------------------------------------------- #
# MODE: render  (watch the student wipe; optional GIF capture)
# --------------------------------------------------------------------------- #
def _caption_frame(arr, text, bar=34):
    """Add a white top bar with `text` to a HxWx3 uint8 frame; return uint8 array."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(arr); W, H = im.size
    canvas = Image.new("RGB", (W, H + bar), (255, 255, 255)); canvas.paste(im, (0, bar))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 8), text, fill=(20, 20, 20), font=font)
    return np.asarray(canvas)


def render_student(args):
    import torch, multiprocessing, ray, time
    from learn import make_env, load_policy

    use_rl = bool(getattr(args, "rl_checkpoint", None))

    _set_env_overrides(args.region)
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False, _node_ip_address="127.0.0.1")
    env = make_env(ENV_NAME, seed=args.seed)

    if use_rl:
        # RLlib policy (e.g. baseline_rl, a teacher) — deterministic mean action
        agent, _ = load_policy(env, "ppo", ENV_NAME, args.rl_checkpoint, seed=args.seed,
                               extra_configs={"num_workers": 0, "num_gpus": 0}, render=True)
        def act(obs):
            return agent.compute_action(obs, explore=False)
    else:
        ckpt = torch.load(args.student, map_location="cpu")
        regions = ckpt["regions"]
        student = _build_student(ckpt["n_ctx"], tuple(ckpt["hidden"]))
        student.load_state_dict(ckpt["state_dict"]); student.eval()
        cvec = context_vec(args.region, regions)
        def act(obs):
            x = torch.tensor(np.concatenate([np.asarray(obs, np.float32), cvec])[None],
                             dtype=torch.float32)
            with torch.no_grad():
                mean, _ = student(x)
            return mean.numpy()[0]

    env.render()                          # MUST come before reset (rebuilds world in GUI)
    b = env.unwrapped if hasattr(env, "unwrapped") else env
    try:                                  # hide debug side panels for a clean shot
        b.bc.configureDebugVisualizer(b.bc.COV_ENABLE_GUI, 0)
    except Exception:
        pass

    def follow():
        tp = b.bc.getLinkState(b.tool, 1, computeForwardKinematics=True,
                               physicsClientId=b.bc._client)[0]
        b.bc.resetDebugVisualizerCamera(cameraDistance=args.cam_dist, cameraYaw=args.cam_yaw,
                                        cameraPitch=args.cam_pitch, cameraTargetPosition=tp)
        return tp

    frames, frame_ep = [], []
    ep_cleared, ep_passes = [], []
    W, H = 720, 540
    for ep in range(args.episodes):
        obs = env.reset(); done = False
        tp = follow()
        t = 0; npass = 0                   # per-episode step counter + completed passes
        while not done:
            obs, _, done, info = env.step(act(obs))
            npass += 1 if info.get("reward_end_pass", 0) > 0 else 0
            tp = follow()
            warming = t < args.skip_start  # skip the startup reach/settle transient
            if args.record_gif:
                if not warming:
                    view = b.bc.computeViewMatrixFromYawPitchRoll(tp, args.cam_dist, args.cam_yaw,
                                                                  args.cam_pitch, 0, 2)
                    proj = b.bc.computeProjectionMatrixFOV(60, W / H, 0.1, 3)
                    img = b.bc.getCameraImage(W, H, view, proj, physicsClientId=b.bc._client)
                    frames.append(np.reshape(img[2], (H, W, 4))[:, :, :3].astype("uint8"))
                    frame_ep.append(ep)
            elif not warming:
                time.sleep(0.02)           # ~real-time once wiping; fast-forward the warmup
            t += 1
        ep_cleared.append(info.get("number_of_targets_cleared", 0)); ep_passes.append(npass)
        print(f"[render] {args.region} ep {ep+1}: cleared={ep_cleared[-1]} passes={npass}")
    env.close()

    if args.record_gif and frames:
        import imageio
        # running means after each episode -> the caption converges to the table value
        run_clr = [float(np.mean(ep_cleared[:k + 1])) for k in range(len(ep_cleared))]
        run_pass = [float(np.mean(ep_passes[:k + 1])) for k in range(len(ep_passes))]
        lab = getattr(args, "label", None)
        if lab:
            out = []
            for i, fr in enumerate(frames):
                ep = frame_ep[i]
                cap = (f"{lab}    ep {ep+1}/{args.episodes}    "
                       f"mean cleared {run_clr[ep]:.1f}/15    mean passes {run_pass[ep]:.2f}")
                out.append(_caption_frame(fr, cap))
            frames = out
        every = max(1, len(frames) // (args.max_frames or len(frames)))
        imageio.mimsave(args.record_gif, frames[::every], fps=args.fps, loop=0)  # loop=0 = play forever
        print(f"[render] saved {len(frames[::every])} frames -> {args.record_gif}  "
              f"(final mean cleared {run_clr[-1]:.2f}, passes {run_pass[-1]:.2f})")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--region", required=True, choices=ALL4_REGIONS)
    c.add_argument("--episodes", type=int, default=250)
    c.add_argument("--out", required=True)
    c.add_argument("--contexts", default="", help="comma-list overriding region order")
    c.add_argument("--style", type=float, default=None,
                   help="append a style scalar to the context (e.g. ours=0, yubik=1) "
                        "for a style-conditioned student that can be EITHER on demand")
    c.add_argument("--seed", type=int, default=1)
    c.set_defaults(func=collect)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True, help="glob, e.g. 'distill_data/*.npz'")
    t.add_argument("--out", default="student.pt")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--beta", type=float, default=1.0, help="weight on KL distill loss")
    t.set_defaults(func=train)

    e = sub.add_parser("eval")
    e.add_argument("--student", required=True)
    e.add_argument("--regions", nargs="+", default=ALL_REGIONS)
    e.add_argument("--episodes", type=int, default=100)
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--force-lo", type=float, default=2.0, help="reward safe-band low (N)")
    e.add_argument("--force-hi", type=float, default=12.0, help="reward safe-band high (N)")
    e.add_argument("--tag", default="student", help="label prefix (e.g. student_ours) so "
                   "multiple students don't collide in behavior_metrics_summary.csv")
    e.add_argument("--style", type=float, default=0.0,
                   help="style value to evaluate a style-conditioned student at "
                        "(matches the --style used in collect; ignored if no style dim)")
    e.set_defaults(func=eval_student)

    r = sub.add_parser("render")
    r.add_argument("--student", default=None, help="torch student .pt (distilled policy)")
    r.add_argument("--rl-checkpoint", default=None,
                   help="instead of --student, render an RLlib checkpoint FILE "
                        "(e.g. baseline_rl or a teacher) for poke-vs-pass comparison")
    r.add_argument("--region", default="forearm_back", choices=ALL4_REGIONS)
    r.add_argument("--episodes", type=int, default=2)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--record-gif", default=None,
                   help="save the rollout as a GIF (e.g. demo_forearm_back.gif)")
    r.add_argument("--label", default=None,
                   help="burn a caption on each frame: this label + running mean "
                        "cleared/passes (converges to the table over many --episodes)")
    r.add_argument("--fps", type=int, default=20)
    r.add_argument("--max-frames", type=int, default=300, help="cap frames in the GIF")
    r.add_argument("--skip-start", type=int, default=40,
                   help="skip the first N steps (reach/settle transient) so the demo "
                        "opens on actual wiping; set 0 to show everything")
    r.add_argument("--cam-dist", type=float, default=1.2)
    r.add_argument("--cam-yaw", type=float, default=150)
    r.add_argument("--cam-pitch", type=float, default=-33)
    r.set_defaults(func=render_student)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

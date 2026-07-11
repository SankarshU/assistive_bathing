#!/usr/bin/env python
"""
make_demos.py — the three demo GIFs, matched to the paper's three claims.  (v4)

  demo_dial.gif                SAME network, forearm_front: gentle (style=0) vs thorough
                               (style=1) side by side — the caregiver dial.
  demo_learned_vs_scripted.gif upperarm_back: distilled policy vs the non-RL scripted sweep.
  demo_whole_arm.gif           ONE network wiping all four regions in sequence.

v4 (after "angles bad / too fast / gentle doesn't clear"):
  * CLOSE-UP follow camera on the tool-skin contact (dist 0.5, yaw 110, pitch -20, FOV 50).
  * REAL-TIME pace: every step captured, 10 fps == the env's 10 Hz.
  * LIVE captions: cleared count ticks up as targets vanish + a FORCE GAUGE with the
    2-12 N therapeutic band shaded — gentleness is visible as the needle staying in-band,
    not as fewer clears.
  * Dial uses best-of-10 episodes per mode so BOTH modes visibly clear well.
Headless TinyRenderer with shadows, 2x supersampled + LANCZOS downscale.

  python make_demos.py                 # all three (~35 min)
  python make_demos.py --only dial
"""
import argparse
import os
import numpy as np
import torch
import imageio
import gym
import learn_bathing  # noqa: F401
from PIL import Image, ImageDraw, ImageFont

from distill import _set_env_overrides, _build_student, context_vec, ENV_NAME
from scripted_baseline import scripted_action, _base

W, H = 720, 540
SS = 2
CAM = dict(dist=0.55, yaw=290, pitch=-25, fov=50)   # far-side view: tool-skin contact unoccluded
STYLE_PT = "student_style_4r_s3.pt"
BAR = 64                                   # caption strip height
F_LO, F_HI, F_MAX = 2.0, 12.0, 20.0        # gauge band + range

try:
    FONT = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    FONT_S = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
except Exception:
    FONT = FONT_S = ImageFont.load_default()


def caption(fr, label, cleared, force):
    """Top strip: '<label>   cleared n/15' + a live force gauge with the 2-12N band."""
    img = Image.new("RGB", (fr.shape[1], fr.shape[0] + BAR), "white")
    img.paste(Image.fromarray(fr), (0, BAR))
    d = ImageDraw.Draw(img)
    d.text((10, 6), f"{label}    cleared {cleared}/15", fill="black", font=FONT)
    # force gauge (right side): 0..F_MAX N, band [2,12] shaded green
    gx0, gx1, gy0, gy1 = img.width - 260, img.width - 14, 12, 34
    d.rectangle([gx0, gy0, gx1, gy1], outline="black", fill=(238, 238, 238))
    bx0 = gx0 + (gx1 - gx0) * F_LO / F_MAX
    bx1 = gx0 + (gx1 - gx0) * F_HI / F_MAX
    d.rectangle([bx0, gy0, bx1, gy1], fill=(200, 235, 200))
    f = min(max(force, 0.0), F_MAX)
    nx = gx0 + (gx1 - gx0) * f / F_MAX
    needle_col = (30, 140, 30) if F_LO <= force <= F_HI else ((200, 30, 30) if force > F_HI else (150, 150, 150))
    d.rectangle([nx - 2, gy0 - 3, nx + 2, gy1 + 3], fill=needle_col)
    d.text((gx0, gy1 + 4), f"force {force:4.1f} N   (band 2-12 N)", fill=needle_col, font=FONT_S)
    return np.asarray(img)


def load_style_student():
    ck = torch.load(STYLE_PT, map_location="cpu")
    net = _build_student(ck["n_ctx"], tuple(ck["hidden"]))
    net.load_state_dict(ck["state_dict"]); net.eval()
    return net, ck["regions"]


def student_act(net, regions, region, style):
    cvec = np.concatenate([context_vec(region, regions), [float(style)]]).astype(np.float32)
    def act(obs, b):
        x = torch.tensor(np.concatenate([np.asarray(obs, np.float32), cvec])[None])
        with torch.no_grad():
            mean, _ = net(x)
        return mean.numpy()[0]
    return act


def scripted_act():
    state = {"phase": 0.0}
    def act(obs, b):
        a, state["phase"] = scripted_action(b, state["phase"], press=0.01, speed=0.04)
        return a
    return act


def run_segment(region, act, label, episodes, keep=1, skip=25, seeds=(1,)):
    """Headless close-up render; capture EVERY step; keep best `keep` episodes by a
    showcase score = cleared + contact-time bonus (an episode that wipes ON the skin films
    well; a wandering one doesn't). Searches across several env seeds for pose variety."""
    _set_env_overrides(region)
    kept = []                                  # (score, cleared, [(frame, cleared_t, force_t)])
    for sd in seeds:
        env = gym.make(ENV_NAME); env.seed(sd)
        b = _base(env)
        for ep in range(episodes):
            obs = env.reset(); done = False; t = 0; info = {}
            rec = []; contact_steps = 0
            while not done:
                obs, _, done, info = env.step(act(obs, b))
                contact_steps += 1 if float(info.get("normal_force", 0.0)) > 0.5 else 0
                if t >= skip:
                    tp = b.bc.getLinkState(b.tool, 1, computeForwardKinematics=True,
                                           physicsClientId=b.bc._client)[0]
                    view = b.bc.computeViewMatrixFromYawPitchRoll(tp, CAM["dist"], CAM["yaw"],
                                                                  CAM["pitch"], 0, 2)
                    proj = b.bc.computeProjectionMatrixFOV(CAM["fov"], W / H, 0.05, 3)
                    img = b.bc.getCameraImage(W * SS, H * SS, view, proj,
                                              shadow=1, lightDirection=[1, -1, 2],
                                              physicsClientId=b.bc._client)
                    big = np.reshape(img[2], (H * SS, W * SS, 4))[:, :, :3].astype("uint8")
                    fr = np.asarray(Image.fromarray(big).resize((W, H), Image.LANCZOS))
                    rec.append((fr, int(info.get("number_of_targets_cleared", 0)),
                                float(info.get("normal_force", 0.0))))
                t += 1
            cleared = info.get("number_of_targets_cleared", 0)
            score = cleared + 3.0 * contact_steps / max(t, 1)   # films well = wipes on-skin
            print(f"  [{label}] seed{sd} ep{ep+1} cleared={cleared} "
                  f"contact={contact_steps/max(t,1):.2f}", flush=True)
            kept.append((score, cleared, rec))
            kept.sort(key=lambda x: -x[0]); kept = kept[:keep]
        env.close()

    frames, shown = [], []
    for _score, cleared, rec in kept:
        shown.append(cleared)
        frames += [caption(fr, label, c, f) for fr, c, f in rec]
    print(f"  [{label}] showing best {keep}: cleared={shown}", flush=True)
    return frames, shown


def side_by_side(fa, fb):
    n = max(len(fa), len(fb))
    fa = fa + [fa[-1]] * (n - len(fa))
    fb = fb + [fb[-1]] * (n - len(fb))
    return [np.hstack([a, b]) for a, b in zip(fa, fb)]


def save(frames, path, fps=10):
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"saved {len(frames)} frames ({len(frames)/fps:.0f}s) -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["dial", "ladder", "arm"], default=None)
    args = ap.parse_args()
    net, regions = load_style_student()

    if args.only in (None, "dial"):
        print("== demo 1: the dial (forearm_front) ==", flush=True)
        g, _ = run_segment("forearm_front", student_act(net, regions, "forearm_front", 0.0),
                           "GENTLE mode (style=0)", episodes=6, seeds=(1, 7, 21))
        t, _ = run_segment("forearm_front", student_act(net, regions, "forearm_front", 1.0),
                           "THOROUGH mode (style=1)", episodes=6, seeds=(1, 7))
        save(side_by_side(g, t), "demo_dial.gif")

    if args.only in (None, "ladder"):
        print("== demo 2: learned vs scripted (upperarm_back) ==", flush=True)
        l, _ = run_segment("upperarm_back", student_act(net, regions, "upperarm_back", 1.0),
                           "LEARNED (one distilled policy)", episodes=8)
        s, _ = run_segment("upperarm_back", scripted_act(),
                           "SCRIPTED (non-RL)", episodes=8)
        save(side_by_side(l, s), "demo_learned_vs_scripted.gif")

    if args.only in (None, "arm"):
        print("== demo 3: one network, whole arm ==", flush=True)
        allf = []
        for reg in regions:
            f, _ = run_segment(reg, student_act(net, regions, reg, 1.0),
                               f"ONE POLICY - {reg}", episodes=6)
            allf += f
        save(allf, "demo_whole_arm.gif")

    print("ALL_DEMOS_DONE", flush=True)


if __name__ == "__main__":
    main()

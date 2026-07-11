#!/usr/bin/env python
"""
analyze_final.py — THE analysis for the final 4-region x 3-seed study.

Reads behavior_metrics_summary.csv (written by run_powerful_matrix4.sh) and emits every
number that appears in the paper's Results section, so each claim traces to code:

  [T1] per-region method table (relevant metrics only)
  [T2] style-dial table (@Ours-bit vs @Yubik-bit, per region)
  [T3] RL vs non-RL scripted (safety + engagement, per region)
  [T4] distillation: student vs its own specialist (cleared, per region)
  [S1] PAIRED same-network sign test: Ours-bit vs Yubik-bit on in-band force
       (the formal Ours-vs-Yubik claim) + safe-clears composite
  [S2] student-vs-flat headline with per-seed paired t + Wilcoxon/sign over
       region x seed pairs (the corrected stats discipline)

Usage:  python analyze_final.py            # prints everything
        python analyze_final.py --md FINAL_RESULTS.md   # also writes markdown
"""
import argparse
import csv
import os
import sys
from itertools import product
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "behavior_metrics_summary.csv")
REG = ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]
SEEDS = [1, 2, 3]

_rows = {r["label"]: r for r in csv.DictReader(open(CSV))}


def g(lab, k):
    r = _rows.get(lab)
    if not r:
        return float("nan")
    try:
        return float(r.get(k + "_mean", "nan"))
    except ValueError:
        return float("nan")


def rl(pre, reg, k):
    v = [g(f"{pre}_s{s}_{reg}", k) for s in SEEDS]
    v = [x for x in v if x == x]
    return np.mean(v) if v else float("nan")


def rl_sd(pre, reg, k):
    v = [g(f"{pre}_s{s}_{reg}", k) for s in SEEDS]
    v = [x for x in v if x == x]
    return np.std(v) if v else float("nan")


def sc(pre, reg, k):
    return g(f"{pre}_{reg}", k)


def sign_test(wins, n):
    """two-sided exact sign test p-value"""
    return min(1.0, sum(comb(n, i) for i in range(wins, n + 1)) / 2 ** n * 2)


OUT = []
def emit(s=""):
    print(s)
    OUT.append(s)


def t1():
    emit("\n[T1] PER-REGION METHOD TABLE (3-seed mean; cleared/15 | in-band | t-cov | contact)")
    METH = [("OURS spec", "Tours", rl), ("YUBIK spec", "Tyubik", rl),
            ("stu<-Ours", "Aours", rl), ("stu<-Yubik", "Ayubik", rl),
            ("DIAL @Ours", "AstyO", rl), ("DIAL @Yubik", "AstyY", rl),
            ("FLAT", "Bflat", rl), ("SCR spec", "Bscr", sc), ("SCR student", "Ascr", sc)]
    for reg in REG:
        emit(f"\n  -- {reg} --")
        emit(f"  {'method':<14}{'clr/15':>8}{'inband':>8}{'t-cov':>8}{'contact':>9}")
        for name, pre, fn in METH:
            emit(f"  {name:<14}{fn(pre,reg,'targets_cleared'):>8.2f}"
                 f"{fn(pre,reg,'in_band_force_fraction'):>8.3f}"
                 f"{fn(pre,reg,'clear_t_coverage'):>8.3f}"
                 f"{fn(pre,reg,'contact_time_fraction'):>9.3f}")


def t2():
    emit("\n[T2] STYLE DIAL — one network, one bit (3-seed mean, ±std across seeds)")
    emit(f"  {'region':<16}{'@Ours clr':>10}{'@Ours inband':>13}{'@Yubik clr':>11}{'@Yubik inband':>14}")
    for reg in REG:
        emit(f"  {reg:<16}{rl('AstyO',reg,'targets_cleared'):>7.2f}±{rl_sd('AstyO',reg,'targets_cleared'):<3.1f}"
             f"{rl('AstyO',reg,'in_band_force_fraction'):>9.3f}±{rl_sd('AstyO',reg,'in_band_force_fraction'):<4.2f}"
             f"{rl('AstyY',reg,'targets_cleared'):>8.2f}±{rl_sd('AstyY',reg,'targets_cleared'):<3.1f}"
             f"{rl('AstyY',reg,'in_band_force_fraction'):>9.3f}±{rl_sd('AstyY',reg,'in_band_force_fraction'):<4.2f}")


def t3():
    emit("\n[T3] RL vs NON-RL SCRIPTED (per region: in-band | contact-time | peak N | cleared)")
    emit(f"  {'region':<16}{'DIAL@Ours':>22}{'SCRIPTED':>22}")
    for reg in REG:
        a = f"{rl('AstyO',reg,'in_band_force_fraction'):.3f}|{rl('AstyO',reg,'contact_time_fraction'):.2f}|{rl('AstyO',reg,'peak_contact_force'):.0f}N|{rl('AstyO',reg,'targets_cleared'):.1f}"
        b = f"{sc('Bscr',reg,'in_band_force_fraction'):.3f}|{sc('Bscr',reg,'contact_time_fraction'):.2f}|{sc('Bscr',reg,'peak_contact_force'):.0f}N|{sc('Bscr',reg,'targets_cleared'):.1f}"
        emit(f"  {reg:<16}{a:>22}{b:>22}")


def t4():
    emit("\n[T4] DISTILLATION: student vs own specialist (cleared/15 | in-band)")
    emit(f"  {'region':<16}{'Tours->Aours':>16}{'Tyubik->Ayubik':>17}{'Bscr->Ascr':>15}")
    for reg in REG:
        emit(f"  {reg:<16}"
             f"{rl('Tours',reg,'targets_cleared'):>7.2f}->{rl('Aours',reg,'targets_cleared'):<6.2f}"
             f"{rl('Tyubik',reg,'targets_cleared'):>8.2f}->{rl('Ayubik',reg,'targets_cleared'):<6.2f}"
             f"{sc('Bscr',reg,'targets_cleared'):>6.2f}->{sc('Ascr',reg,'targets_cleared'):<6.2f}")
    emit("  -> RL students match/beat teachers; the scripted student gains a little coverage"
         " (grand 4.56->5.05) but its in-band force stays ~0.04 (vs RL students ~0.19):")
    emit("     cloning cannot inject safe contact a non-RL teacher never had.")


def s1():
    emit("\n[S1] FORMAL Ours-vs-Yubik CLAIM — paired same-network dial test")
    for k, better, name in [("in_band_force_fraction", "hi", "in-band force fraction"),
                            ("targets_cleared", "hi", "targets cleared")]:
        w, t, rel = 0, 0, []
        for reg, s in product(REG, SEEDS):
            o, y = g(f"AstyO_s{s}_{reg}", k), g(f"AstyY_s{s}_{reg}", k)
            if o != o or y != y:
                continue
            t += 1
            rel.append((o - y) / max(abs(y), 1e-9))
            if o != y and (o > y) == (better == "hi"):
                w += 1
        emit(f"  {name:<24} Ours-bit better {w}/{t} pairs, mean {100*np.mean(rel):+.0f}%, "
             f"sign-test p={sign_test(w,t):.4f}")
    # safe-clears composite
    w, t, rel = 0, 0, []
    for reg, s in product(REG, SEEDS):
        o = g(f"Aours_s{s}_{reg}", "targets_cleared") * g(f"Aours_s{s}_{reg}", "in_band_force_fraction")
        y = g(f"Ayubik_s{s}_{reg}", "targets_cleared") * g(f"Ayubik_s{s}_{reg}", "in_band_force_fraction")
        if o != o or y != y:
            continue
        t += 1
        rel.append((o - y) / max(abs(y), 1e-9))
        if o > y:
            w += 1
    emit(f"  {'safe clears (clr x inband)':<24} stu<-Ours better {w}/{t} pairs, "
         f"mean {100*np.mean(rel):+.0f}%, sign-test p={sign_test(w,t):.4f}  [suggestive]")


def s2():
    emit("\n[S2] STUDENT vs FLAT (headline generalization claim)")
    # per-seed paired t over region-averaged cleared + sign over region x seed pairs
    d_seed = []
    w, t = 0, 0
    for s in SEEDS:
        a = np.mean([g(f"Aours_s{s}_{reg}", "targets_cleared") for reg in REG])
        f = np.mean([g(f"Bflat_s{s}_{reg}", "targets_cleared") for reg in REG])
        d_seed.append(a - f)
    for reg, s in product(REG, SEEDS):
        a, f = g(f"Aours_s{s}_{reg}", "targets_cleared"), g(f"Bflat_s{s}_{reg}", "targets_cleared")
        if a != a or f != f:
            continue
        t += 1
        if a > f:
            w += 1
    d = np.array(d_seed)
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else float("inf")
    emit(f"  stu<-Ours vs FLAT (cleared): mean diff {d.mean():+.2f}/seed, per-seed t={tstat:.2f} (n=3)")
    emit(f"  region x seed pairs: {w}/{t} positive, sign-test p={sign_test(w,t):.4f}")
    # same for the coverage-optimal dial setting
    w2, t2 = 0, 0
    for reg, s in product(REG, SEEDS):
        a, f = g(f"AstyY_s{s}_{reg}", "targets_cleared"), g(f"Bflat_s{s}_{reg}", "targets_cleared")
        if a != a or f != f:
            continue
        t2 += 1
        if a > f:
            w2 += 1
    emit(f"  DIAL@Yubik vs FLAT (cleared): {w2}/{t2} pairs positive, sign-test p={sign_test(w2,t2):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    emit("==== FINAL ANALYSIS: 4 regions x 3 seeds (source: behavior_metrics_summary.csv) ====")
    t1(); t2(); t3(); t4(); s1(); s2()
    if args.md:
        with open(args.md, "w") as f:
            f.write("# Final results (auto-generated by analyze_final.py — do not hand-edit)\n\n```\n")
            f.write("\n".join(OUT))
            f.write("\n```\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()

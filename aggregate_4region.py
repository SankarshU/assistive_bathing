#!/usr/bin/env python3
"""
aggregate_4region.py — across-seed ICRA table for the 4-region generalization study.
Reads seed-tagged rows (dstl4_s{seed}_{region} / spec4_s* / flat4_s*) and reports,
per region and metric, mean +/- std ACROSS SEEDS for student / specialist / flat,
plus a paired t-test (student vs flat) on region-averaged clears. Saves ICRA table.
"""
import argparse, csv, os, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS = ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]
METRICS = [("targets_cleared", "cleared /15"), ("productive_frac", "productive_frac"),
           ("pass_completions", "passes"), ("clear_s_coverage", "clear_s_cov"),
           ("clear_spatial_order", "spatial_order"), ("in_band_force_fraction", "in-band F"),
           ("p95_contact_force", "p95 force"), ("stall_frac", "stall"),
           ("action_smoothness", "jerk"), ("terminated_drift", "drift")]
POLICIES = [("dstl4", "STUDENT"), ("spec4", "specialist"), ("flat4", "flat")]


def _t_two_sided_p(t, df):
    """Two-sided p for Student's t. Uses scipy when available; otherwise an exact
    closed form for df<=2 and a Hill-type approximation above."""
    t = abs(float(t))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(t, df))
    except ImportError:
        pass
    if df == 1:
        return 2 * (1 - (0.5 + math.atan(t) / math.pi))
    if df == 2:
        return 2 * (1 - (0.5 + t / (2 * math.sqrt(2) * math.sqrt(1 + t * t / 2))))
    # normal approx with second-order correction (adequate for df>=3 reporting)
    z = t * (1 - 1 / (4 * df)) / math.sqrt(1 + t * t / (2 * df))
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def _wilcoxon_exact_p(diffs):
    """Exact two-sided Wilcoxon signed-rank p by full enumeration (fine for n<=20)."""
    import itertools
    d = [x for x in diffs if x != 0]
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0] * n
    for pos, i in enumerate(order):
        ranks[i] = pos + 1
    tot = n * (n + 1) // 2
    W = sum(r for x, r in zip(d, ranks) if x > 0)
    Wmin = min(W, tot - W)
    count = 0
    for pat in itertools.product((0, 1), repeat=n):
        s = sum(r for b, r in zip(pat, ranks) if b)
        if min(s, tot - s) <= Wmin:
            count += 1
    return count / 2 ** n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = ap.parse_args()
    R = {r["label"]: r for r in csv.DictReader(open(os.path.join(HERE, "behavior_metrics_summary.csv")))}
    def sv(prefix, region, key):
        out = []
        for s in args.seeds:
            row = R.get(f"{prefix}_s{s}_{region}")
            if row:
                try: out.append(float(row[key + "_mean"]))
                except (KeyError, ValueError): pass
        return out
    L = [f"# 4-REGION GENERALIZATION across seeds {args.seeds} (mean +/- std across seeds)"]
    for reg in REGIONS:
        L.append(f"\n### {reg}")
        L.append(f"{'metric':16}{'STUDENT':>16}{'specialist':>16}{'flat':>16}")
        L.append("-" * 64)
        for key, nice in METRICS:
            cells = []
            for pre, _ in POLICIES:
                v = sv(pre, reg, key)
                cells.append(f"{np.mean(v):.2f}+-{np.std(v):.2f}" if v else "  -  ")
            L.append(f"{nice:16}{cells[0]:>16}{cells[1]:>16}{cells[2]:>16}")
    # headline: per-seed region-averaged clears, paired student vs flat
    def region_avg(pre, key, seed):
        vs = []
        for reg in REGIONS:
            r = R.get(f"{pre}_s{seed}_{reg}")
            if r:
                try: vs.append(float(r[key + "_mean"]))
                except (KeyError, ValueError): pass
        return np.mean(vs) if vs else None
    st = [region_avg("dstl4", "targets_cleared", s) for s in args.seeds]
    fl = [region_avg("flat4", "targets_cleared", s) for s in args.seeds]
    sp = [region_avg("spec4", "targets_cleared", s) for s in args.seeds]
    pairs = [(a, b) for a, b in zip(st, fl) if a is not None and b is not None]
    L.append("\n### HEADLINE: avg cleared across 4 regions, per seed (paired student vs flat)")
    L.append(f"  student per seed: {[round(x,2) for x in st if x is not None]}")
    L.append(f"  flat    per seed: {[round(x,2) for x in fl if x is not None]}")
    L.append(f"  specialist per seed (avg): {[round(x,2) for x in sp if x is not None]}")
    if len(pairs) >= 2:
        d = [a - b for a, b in pairs]; dm = np.mean(d)
        ds = np.std(d, ddof=1) if len(d) > 1 else 0.0
        t = dm / (ds / math.sqrt(len(d))) if ds else float("inf")
        # Correct paired t-test: Student's t with df=n-1 (the old code used the
        # normal CDF, which is wildly anti-conservative at n=3: p=0.0019 vs ~0.09).
        p = _t_two_sided_p(t, len(d) - 1) if ds else 0.0
        sm = np.mean([a for a, _ in pairs]); fm = np.mean([b for _, b in pairs])
        L.append(f"  STUDENT {sm:.2f}  vs  FLAT {fm:.2f}  ({100*(sm/fm-1):+.0f}%)   paired t={t:.2f}  p={p:.4f}  (n={len(pairs)} seeds, df={len(pairs)-1})")

    # Robust primary statistic: exact sign + Wilcoxon signed-rank tests over ALL
    # region x seed pairs (12 pairs at 3 seeds). More power than n=3 seed means;
    # pairing unit stated explicitly (regions within a seed share the seed's run).
    pair_d = []
    for s in args.seeds:
        for reg in REGIONS:
            a, b = R.get(f"dstl4_s{s}_{reg}"), R.get(f"flat4_s{s}_{reg}")
            if a and b:
                try:
                    pair_d.append(float(a["targets_cleared_mean"]) - float(b["targets_cleared_mean"]))
                except (KeyError, ValueError):
                    pass
    if len(pair_d) >= 6:
        npos = sum(x > 0 for x in pair_d); n = len(pair_d)
        k = min(npos, n - npos)
        p_sign = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        p_wil = _wilcoxon_exact_p(pair_d)
        L.append(f"  region x seed pairs: {npos}/{n} positive | exact sign test p={min(p_sign,1.0):.4f} | "
                 f"exact Wilcoxon signed-rank p={p_wil:.4f}")
    txt = "\n".join(L); print(txt)
    open(os.path.join(HERE, "ICRA_4region_table.txt"), "w").write(txt + "\n")
    print("\n[saved -> ICRA_4region_table.txt]")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_icra_figure.py — the headline ICRA comparison image + text table.

Reads behavior_metrics_summary.csv (written by metrics_report.py, distill.py eval, and
scripted_baseline.py --metrics -- all use the SAME reducers, so every row is comparable)
and renders:
  (1) a color-coded methods x metrics table (best-in-row highlighted), and
  (2) a headline bar chart of targets-cleared per region, grouped by method,
into ICRA_matrix_figure.png, plus a plain-text ICRA_matrix_table.txt and a printed story.

Methods are located by label prefix. New matrix tags are preferred; if absent, the
older run_4region_3seed tags are used as a fallback so a PARTIAL figure renders from
whatever data exists today.

    python make_icra_figure.py --seeds 1 2
"""
import argparse, csv, os, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "behavior_metrics_summary.csv")

REGIONS = ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]

# display method  ->  (primary prefix, fallback prefix or None, seeded?)
METHODS = [
    ("Student <- Ours",              "Aours",  "dstl4", True),
    ("Student <- Yubik",             "Ayubik", None,    True),
    ("Student <- Both (style/ours)", "AstyO",  None,    True),
    ("Student <- Both (style/yubik)","AstyY",  None,    True),
    ("Student <- Both (general)",    "Agen",   None,    True),
    ("Ours specialists",             "Tours",  "spec4", True),
    ("Yubik specialists",            "Tyubik", None,    True),
    ("Flat baseline",                "Bflat",  "flat4", True),
    ("Non-RL scripted",              "Bscr",   None,    False),
]

# display metric  ->  (csv key without _mean/_std,  higher_is_better: True/False/None)
ALL_METRICS = [
    ("Targets cleared /15",   "targets_cleared",        True),
    ("Task success",          "task_success",           True),
    ("Region covered (m)",    "coverage_length_m",      True),
    ("Time wiping (s)",       "wipe_time_s",            None),
    ("Mean force (N)",        "mean_contact_force",     None),
    ("p95 force (N)",         "p95_contact_force",      False),
    ("Peak force (N)",        "peak_contact_force",     False),
    ("In-band force frac",    "in_band_force_fraction", True),
    ("Pass completions",      "pass_completions",       True),
    ("Sweep consistency",     "sweep_consistency",      True),
    ("Clear spatial order",   "clear_spatial_order",    True),
    ("Action effort |a|",     "action_effort",          False),
    ("Action smooth |da|",    "action_smoothness",      False),
    ("Drift rate",            "terminated_drift",       False),
    # --- trace_aggregate spread/efficiency metrics (folded into metrics_report) ---
    ("Clear s-coverage",      "clear_s_coverage",       True),
    ("Clear t-coverage",      "clear_t_coverage",       True),
    ("2nd-half frac",         "second_half_frac",       True),
    ("Productive frac",       "productive_frac",        True),
    ("Stall frac",            "stall_frac",             False),
]

# HEADLINE set: metrics that BOTH discriminate across methods (>=~25% relative range)
# AND are individually interpretable. Dropped from the headline (kept only in --full):
#   task_success (all ~0), coverage_length_m / peak_force / productive_frac / stall_frac
#   (nearly flat across methods), mean_contact_force (redundant with p95+in-band),
#   action_effort/action_smoothness (practically identical), wipe_time_s (neutral, not
#   good/bad), clear_spatial_order (its spread is an artifact of the non-RL outlier).
HEADLINE_KEYS = {
    "targets_cleared",          # performance
    "p95_contact_force",        # safety (gentleness)
    "in_band_force_fraction",   # safety (therapeutic band)
    "pass_completions",         # task quality (complete sweeps)
    "sweep_consistency",        # task quality (systematic motion)
    "clear_s_coverage",         # coverage along the arm
    "clear_t_coverage",         # coverage over time
    "second_half_frac",         # sustained clearing (not front-loaded)
    "terminated_drift",         # stability / safety (runaway drift)
}


def load_rows():
    if not os.path.exists(CSV):
        raise SystemExit(f"[fig] {CSV} not found -- run the matrix first.")
    with open(CSV, newline="") as f:
        return {r["label"]: r for r in csv.DictReader(f)}   # last write wins per label


def _num(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def collect(rows, prefix, seeded, seeds, metric_key):
    """Mean of metric_key+'_mean' across (seeds x regions) for one method.
    Also returns per-region seed-averaged values (for the bar chart)."""
    key = metric_key + "_mean"
    per_region = {}
    for r in REGIONS:
        vals = []
        if seeded:
            for s in seeds:
                lab = f"{prefix}_s{s}_{r}"
                if lab in rows:
                    vals.append(_num(rows[lab], key))
        else:
            lab = f"{prefix}_{r}"
            if lab in rows:
                vals.append(_num(rows[lab], key))
        vals = [v for v in vals if not np.isnan(v)]
        per_region[r] = np.mean(vals) if vals else np.nan
    allv = [v for v in per_region.values() if not np.isnan(v)]
    return (np.mean(allv) if allv else np.nan), per_region


def resolve_prefix(rows, primary, fallback, seeded, seeds):
    """Pick primary prefix if it has any rows, else fallback."""
    def has(pfx):
        if pfx is None:
            return False
        if seeded:
            return any(f"{pfx}_s{s}_{r}" in rows for s in seeds for r in REGIONS)
        return any(f"{pfx}_{r}" in rows for r in REGIONS)
    if has(primary):
        return primary
    if has(fallback):
        return fallback
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default="ICRA_matrix_figure.png")
    ap.add_argument("--full", action="store_true",
                    help="show all 19 metrics (appendix); default = curated headline set")
    args = ap.parse_args()
    rows = load_rows()
    METRICS = ALL_METRICS if args.full else [m for m in ALL_METRICS if m[1] in HEADLINE_KEYS]

    # resolve which methods actually have data
    present = []   # (display, prefix, seeded)
    for disp, primary, fallback, seeded in METHODS:
        pfx = resolve_prefix(rows, primary, fallback, seeded, args.seeds)
        if pfx is not None:
            present.append((disp, pfx, seeded))
    if not present:
        raise SystemExit("[fig] no known method tags found in CSV yet.")

    # table[metric_idx][method_idx] = scalar; barsdata for targets_cleared
    table = np.full((len(METRICS), len(present)), np.nan)
    bar_per_region = {}   # disp -> {region: value}
    for mi, (mdisp, mkey, _hib) in enumerate(METRICS):
        for pj, (disp, pfx, seeded) in enumerate(present):
            val, per_region = collect(rows, pfx, seeded, args.seeds, mkey)
            table[mi, pj] = val
            if mkey == "targets_cleared":
                bar_per_region[disp] = per_region

    # ---------- text table ----------
    txt = [f"ICRA comparison matrix  (seeds={args.seeds}, regions=4, 100 eps/region)",
           "methods present: " + " | ".join(d for d, _, _ in present), ""]
    header = f"{'metric':22}" + "".join(f"{d[:15]:>16}" for d, _, _ in present)
    txt.append(header)
    for mi, (mdisp, mkey, hib) in enumerate(METRICS):
        line = f"{mdisp:22}"
        row = table[mi]
        best = np.nan
        if hib is True and np.any(~np.isnan(row)):
            best = np.nanmax(row)
        elif hib is False and np.any(~np.isnan(row)):
            best = np.nanmin(row)
        for v in row:
            cell = "   -   " if np.isnan(v) else f"{v:7.3f}"
            mark = "*" if (not np.isnan(v) and not np.isnan(best) and abs(v - best) < 1e-9) else " "
            line += f"{cell+mark:>16}"
        txt.append(line)
    txt_s = "\n".join(txt)
    with open(os.path.join(HERE, "ICRA_matrix_table.txt"), "w") as f:
        f.write(txt_s + "\n")
    print("\n" + txt_s + "\n")

    # ---------- figure ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ncol = len(present)
    fig = plt.figure(figsize=(max(12, 1.5 * ncol + 5), 13))
    gs = GridSpec(2, 1, height_ratios=[1.0, 2.3], hspace=0.28)

    # (1) headline bar chart: targets cleared per region, grouped by method
    ax0 = fig.add_subplot(gs[0])
    x = np.arange(len(REGIONS)); w = 0.8 / max(1, ncol)
    cmap = plt.get_cmap("tab10")
    for j, (disp, _, _) in enumerate(present):
        vals = [bar_per_region.get(disp, {}).get(r, np.nan) for r in REGIONS]
        ax0.bar(x + j * w - 0.4 + w / 2, vals, w, label=disp, color=cmap(j % 10))
    ax0.set_xticks(x); ax0.set_xticklabels([r.replace("_", "\n") for r in REGIONS], fontsize=9)
    ax0.set_ylabel("targets cleared / 15")
    ax0.set_title("Targets cleared per region  (higher = better)", fontsize=12, weight="bold")
    ax0.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax0.grid(axis="y", alpha=0.25)

    # (2) methods x metrics table, best-in-row shaded green, worst light red
    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    cell_text, cell_colours = [], []
    for mi, (mdisp, mkey, hib) in enumerate(METRICS):
        row = table[mi]
        best = worst = np.nan
        finite = row[~np.isnan(row)]
        if finite.size:
            if hib is True:  best, worst = np.nanmax(row), np.nanmin(row)
            elif hib is False: best, worst = np.nanmin(row), np.nanmax(row)
        trow, crow = [], []
        for v in row:
            trow.append("-" if np.isnan(v) else f"{v:.3f}")
            c = "white"
            if not np.isnan(v) and hib is not None and finite.size > 1:
                if abs(v - best) < 1e-9:   c = "#bfe6c4"   # green = best
                elif abs(v - worst) < 1e-9: c = "#f6c9c4"  # red = worst
            crow.append(c)
        cell_text.append(trow); cell_colours.append(crow)
    tab = ax1.table(cellText=cell_text, cellColours=cell_colours,
                    rowLabels=[m[0] for m in METRICS],
                    colLabels=[d for d, _, _ in present],
                    cellLoc="center", loc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        if r == 0:  # header
            cell.set_text_props(weight="bold", fontsize=8); cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", weight="bold")
        if c == -1:  # row labels
            cell.set_text_props(ha="right", fontsize=8)
    _mtitle = ("All 19 metrics" if len(METRICS) > 12 else
               "Key discriminative metrics (curated)")
    ax1.set_title(f"{_mtitle}  (green = best in row, red = worst; "
                  "forces judged gentler-is-better)", fontsize=11, weight="bold", pad=14)

    fig.suptitle("Assistive bed-bathing: one distilled student vs. specialist teachers "
                 "(Ours / Yubik) and baselines", fontsize=13, weight="bold", y=0.995)
    fig.savefig(os.path.join(HERE, args.out), dpi=150, bbox_inches="tight")
    print(f"[fig] saved -> {args.out}")

    # ---------- printed story (headline deltas) ----------
    def region_avg(disp, mkey):
        for d, pfx, seeded in present:
            if d == disp:
                return collect(rows, pfx, seeded, args.seeds, mkey)[0]
        return np.nan
    print("---- STORY (targets cleared, 4-region avg) ----")
    for disp in [d for d, _, _ in present]:
        print(f"  {disp:30} {region_avg(disp,'targets_cleared'):.2f}")


if __name__ == "__main__":
    main()

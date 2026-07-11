#!/usr/bin/env python3
"""
make_paper_figs.py — publication-quality (vector) figures + flagship-table numbers.

Replaces the old rasterized methods-x-metrics table image. Produces:
  paper/fig_tradeoff.pdf   coverage <-> safety trade-off scatter (the results teaser);
                           each method a point, the CCP dial drawn as an arrow.
  paper/fig_coverage.pdf   grouped bar: targets cleared per region, key methods.
  prints the flagship-table grand means (methods x metrics) for the LaTeX table.

All numbers come from behavior_metrics_summary.csv (same reducers as the paper).
    python make_paper_figs.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "behavior_metrics_summary.csv")
PAPER = os.path.join(HERE, "paper")
REG = ["forearm_back", "forearm_front", "upperarm_back", "upperarm_front"]
SEEDS = [1, 2, 3]
rows = {r["label"]: r for r in csv.DictReader(open(CSV))}

# paper-name -> (csv prefix, seeded?)
METHODS = [
    ("Comfort spec.",      "Tours",  True),
    ("Contact-cnt spec.",  "Tyubik", True),
    ("Student<-Comfort",   "Aours",  True),
    ("Student<-Contact",   "Ayubik", True),
    ("CCP @Gentle",        "AstyO",  True),
    ("CCP @Thorough",      "AstyY",  True),
    ("Flat",               "Bflat",  True),
    ("Scripted",           "Bscr",   False),
]
# metric -> (csv key, higher_is_better)
METRICS = [
    ("Cleared/15",   "targets_cleared",        True),
    ("In-band",      "in_band_force_fraction", True),
    ("Contact-t",    "contact_time_fraction",  True),
    ("Sweep",        "sweep_consistency",      True),
    ("s-cover",      "clear_s_coverage",       True),
    ("Peak N",       "peak_contact_force",     False),
]


def g(lab, k):
    r = rows.get(lab)
    try:
        return float(r[k + "_mean"])
    except Exception:
        return np.nan


def grand(prefix, seeded, key):
    vals = []
    for rg in REG:
        if seeded:
            v = [g(f"{prefix}_s{s}_{rg}", key) for s in SEEDS]
            v = [x for x in v if x == x]
            vals.append(np.mean(v) if v else np.nan)
        else:
            vals.append(g(f"{prefix}_{rg}", key))
    vals = [x for x in vals if x == x]
    return np.mean(vals) if vals else np.nan


def per_region(prefix, seeded, key):
    out = []
    for rg in REG:
        if seeded:
            v = [g(f"{prefix}_s{s}_{rg}", key) for s in SEEDS]
            v = [x for x in v if x == x]
            out.append(np.mean(v) if v else np.nan)
        else:
            out.append(g(f"{prefix}_{rg}", key))
    return out


# ---------- flagship table numbers (printed; hand-placed into the .tex) ----------
print("\n=== FLAGSHIP TABLE (grand mean over 4 regions) ===")
hdr = f"{'method':18}" + "".join(f"{m[0]:>11}" for m in METRICS)
print(hdr)
data = {}
for disp, pfx, seeded in METHODS:
    data[disp] = {mk: grand(pfx, seeded, mk) for _, mk, _ in METRICS}
    line = f"{disp:18}" + "".join(f"{data[disp][mk]:>11.3f}" for _, mk, _ in METRICS)
    print(line)
# best per column (for bolding in LaTeX)
print("\nBEST per column (bold these in LaTeX):")
for name, mk, hib in METRICS:
    col = {d: data[d][mk] for d in data if data[d][mk] == data[d][mk]}
    best = (max if hib else min)(col, key=col.get)
    print(f"  {name:12} -> {best}  ({col[best]:.3f})")

# ---------- Figure A: coverage<->safety trade-off scatter (legend, no overlaps) ----------
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
xs = {d: grand(p, s, "targets_cleared") for d, p, s in METHODS}
ys = {d: grand(p, s, "in_band_force_fraction") for d, p, s in METHODS}
# (display, marker, color, size) — CCP stars stand out; baselines squares; others circles
STYLE = {
    "CCP @Gentle":       ("*", "#1b7837", 260),
    "CCP @Thorough":     ("*", "#762a83", 260),
    "Comfort spec.":     ("o", "#7fbf7b", 60),
    "Contact-cnt spec.": ("o", "#af8dc3", 60),
    "Student<-Comfort":  ("D", "#1b7837", 45),
    "Student<-Contact":  ("D", "#762a83", 45),
    "Flat":              ("s", "#b35806", 70),
    "Scripted":          ("s", "#b2182b", 70),
}
for disp, pfx, seeded in METHODS:
    m, c, sz = STYLE[disp]
    ax.scatter(xs[disp], ys[disp], s=sz, c=c, marker=m, zorder=3,
               edgecolors="k", linewidths=0.5, label=disp)
# dial arrow Thorough -> Gentle
ax.annotate("", xy=(xs["CCP @Gentle"], ys["CCP @Gentle"]),
            xytext=(xs["CCP @Thorough"], ys["CCP @Thorough"]),
            arrowprops=dict(arrowstyle="->", color="#542788", lw=1.8), zorder=2)
ax.text((xs["CCP @Gentle"]+xs["CCP @Thorough"])/2 - 0.02,
        (ys["CCP @Gentle"]+ys["CCP @Thorough"])/2 + 0.006,
        "comfort dial", color="#542788", fontsize=7.5, ha="center", style="italic")
ax.set_xlabel("Targets cleared / 15  (coverage $\\rightarrow$)")
ax.set_ylabel("In-band force fraction  (safety $\\rightarrow$)")
ax.grid(alpha=0.25)
ax.set_ylim(0.0, 0.23)
ax.legend(fontsize=6.0, ncol=2, loc="lower center", framealpha=0.92,
          handletextpad=0.2, columnspacing=0.8, borderpad=0.3)
ax.set_title("One policy, two behaviours: the comfort dial spans\nthe safety$\\leftrightarrow$coverage frontier; baselines are dominated",
             fontsize=8.0, weight="bold")
fig.tight_layout()
fig.savefig(os.path.join(PAPER, "fig_tradeoff.pdf"), bbox_inches="tight")
print("\n[fig] paper/fig_tradeoff.pdf")

# ---------- Figure B: grouped bar, targets cleared per region ----------
key_methods = ["CCP @Thorough", "CCP @Gentle", "Contact-cnt spec.", "Flat", "Scripted"]
fig2, ax2 = plt.subplots(figsize=(3.4, 2.4))
x = np.arange(len(REG)); w = 0.8 / len(key_methods)
pal = ["#762a83", "#1b7837", "#4d4d4d", "#b35806", "#b2182b"]
for j, disp in enumerate(key_methods):
    pfx, seeded = dict((d, (p, s)) for d, p, s in METHODS)[disp]
    vals = per_region(pfx, seeded, "targets_cleared")
    ax2.bar(x + j * w - 0.4 + w / 2, vals, w, label=disp, color=pal[j])
ax2.set_xticks(x)
ax2.set_xticklabels(["fore\nback", "fore\nfront", "upper\nback", "upper\nfront"], fontsize=7)
ax2.set_ylabel("Targets cleared / 15")
ax2.legend(fontsize=6, ncol=2, loc="upper right", framealpha=0.9)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title("Coverage per region", fontsize=8.5, weight="bold")
fig2.tight_layout()
fig2.savefig(os.path.join(PAPER, "fig_coverage.pdf"), bbox_inches="tight")
print("[fig] paper/fig_coverage.pdf")

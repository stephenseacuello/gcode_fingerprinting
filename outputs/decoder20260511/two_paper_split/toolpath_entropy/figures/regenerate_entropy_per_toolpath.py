#!/usr/bin/env python3
"""Regenerate entropy_per_toolpath.pdf with a legible layout (no title/label overlap).

Per-toolpath empirical source entropy decomposed by prediction head, across the nine
operation classes grouped by toolpath family (face / pocket / adaptive). Source:
audit/per_head_per_opclass_entropy_5fold.json (mean_entropy_matrix, heads x op_classes).

Layout fix (replaces the lost original generator whose centered title overprinted the
family-group labels): the embedded title is dropped (the LaTeX caption states the content),
the head legend sits above the axes, and the FACE/POCKET/ADAPTIVE family labels are drawn
in a dedicated row below the rotated op-class ticks with light separators between families.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.abspath(os.path.join(HERE, "../../../audit"))
SRC = os.path.join(AUDIT, "per_head_per_opclass_entropy_5fold.json")
OUT = os.path.join(HERE, "entropy_per_toolpath.pdf")

d = json.load(open(SRC))
heads = d["heads"]                      # token,type,command,param_type,sign,digit
ocs = d["op_classes"]
M = np.array(d["mean_entropy_matrix"])  # heads x op_classes
SD = np.array(d["sd_entropy_matrix"])

# Order op-classes by family, base->150025->damage within each family.
families = [("FACE", ["face", "face150025", "damageface"]),
            ("POCKET", ["pocket", "pocket150025", "damagepocket"]),
            ("ADAPTIVE", ["adaptive", "adaptive150025", "damageadaptive"])]
order, fam_spans, lbls = [], [], []
oi = {o: i for i, o in enumerate(ocs)}
pos = 0
for fam, members in families:
    members = [m for m in members if m in oi]
    start = pos
    for m in members:
        order.append(oi[m]); lbls.append(m); pos += 1
    fam_spans.append((fam, start, pos - 1))

M = M[:, order]; SD = SD[:, order]
nH, nO = M.shape
x = np.arange(nO)
bw = 0.8 / nH
colors = plt.get_cmap("tab10").colors
head_disp = {"token": "Token", "type": "Type", "command": "Command",
             "param_type": "Param-type", "sign": "Sign", "digit": "Digit"}

fig, ax = plt.subplots(figsize=(11.5, 4.6))
for h in range(nH):
    ax.bar(x + (h - (nH - 1) / 2) * bw, M[h], bw, yerr=SD[h],
           label=head_disp.get(heads[h], heads[h]), color=colors[h % 10],
           error_kw=dict(lw=0.6, capsize=1.5, alpha=0.5))

ax.set_ylabel("Source entropy $H$ (bits)", fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(lbls, rotation=40, ha="right", fontsize=8.5)
ax.set_xlim(-0.6, nO - 0.4)
ax.set_ylim(0, max(7.0, M.max() * 1.12))
ax.grid(axis="y", ls=":", alpha=0.4)
ax.legend(ncol=nH, fontsize=9, loc="upper center",
          bbox_to_anchor=(0.5, 1.12), frameon=False, columnspacing=1.3, handlelength=1.2)

# Family-group labels + separators in a dedicated row below the tick labels.
for fam, s, e in fam_spans:
    ax.text((s + e) / 2.0, -0.30, fam, ha="center", va="top",
            transform=ax.get_xaxis_transform(), fontsize=10, fontweight="bold")
for _, s, e in fam_spans[:-1]:
    ax.axvline(e + 0.5, color="0.7", lw=0.8, ls="--")

fig.subplots_adjust(top=0.88, bottom=0.30, left=0.07, right=0.99)
fig.savefig(OUT)
print("wrote", OUT, "| heads", heads, "| op-classes(ordered)", lbls)

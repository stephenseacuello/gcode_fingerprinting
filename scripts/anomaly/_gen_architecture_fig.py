"""Regenerate the anomaly paper's architecture overview figure.

Replaces the original one-off figure (source lost) whose encoder-memory
annotation predated the d_model correction: the latent is z in R^{L x 256}
(encoder d_model = 256), not L x 128. All other content matches the original.

Output: outputs/anomaly20260319/anomalypaper/figures/architecture_overview.pdf
(copy into two_paper_split/anomaly_detection/figures/ to update the paper).
"""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/anomaly20260319/anomalypaper/figures')
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE_BG, BLUE_BOX = '#cfe3f7', '#a8ccee'
GREEN_BG, GREEN_BOX = '#c9efd2', '#8fdca4'
ORANGE = '#fcdcb0'
RED = '#f9c2c2'
GRAY = '#e8edf2'
EDGE = '#4a6a8a'

fig, ax = plt.subplots(figsize=(8.64, 4.86))
ax.set_xlim(0, 100)
ax.set_ylim(0, 56)
ax.axis('off')


def box(x, y, w, h, fc, lw=0.8, r=0.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=EDGE, lw=lw, zorder=2)
    ax.add_patch(p)
    return p


def label(x, y, s, size=5.2, weight='normal', color='black', style='normal', ha='center'):
    ax.text(x, y, s, ha=ha, va='center', fontsize=size, weight=weight,
            color=color, style=style, zorder=3)


def arrow(x1, y1, x2, y2, style='-|>', lw=0.9, cs='arc3,rad=0.0'):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, lw=lw,
                        color='#333333', connectionstyle=cs,
                        mutation_scale=8, zorder=1)
    ax.add_patch(a)


# --- sensor stack (far left) ---
for i in range(6):
    box(1.5, 40.5 + i * 1.7, 8, 1.3, GRAY, lw=0.5, r=0.2)
label(5.5, 37.8, '6 × Arduino Nano\n33 BLE Sense', size=4.8, weight='bold')
label(5.5, 34.6, '110 channels\n7 modalities', size=4.4)

# --- encoder block ---
box(14, 27, 26, 26, BLUE_BG)
label(27, 50.5, 'MM-DTAE-LSTM Encoder', size=6.2, weight='bold')
label(27, 48.3, '(5.1M params, frozen)', size=4.8)
for yy, txt in [(43.2, '7 Modality Encoders'), (37.4, 'Cross-Modal Fusion'), (31.6, 'LSTM Bridge')]:
    box(18, yy, 18, 3.6, BLUE_BOX)
    label(27, yy + 1.8, txt, size=5.2)
    if yy > 31.6:
        arrow(27, yy, 27, yy - 2.2)

# classification head (below encoder)
box(18, 19.5, 18, 4.2, BLUE_BOX)
label(27, 21.6, r'Classification Head  $\hat{c} \in \{1,\dots,9\}$', size=5.2)
arrow(27, 31.6, 27, 23.9)

# sensors -> encoder
arrow(9.7, 45.5, 13.8, 45.5)

# --- encoder memory annotation (corrected: L x 256) ---
arrow(40.2, 40, 51.8, 44.5, cs='arc3,rad=-0.25')
label(45.5, 44.3, r'$\mathbf{z} \in \mathbb{R}^{L \times 256}$', size=5.4, style='italic')
label(45.5, 42.3, 'Encoder Memory', size=4.6, style='italic')

# --- decoder block ---
box(52, 26, 26, 27, GREEN_BG)
label(65, 50.5, 'SensorMultiHeadDecoder', size=6.2, weight='bold')
label(65, 48.3, '(22.1M params)', size=4.8)
for yy, txt in [(43.2, 'Cross-Attention to z'),
                (36.6, '8-layer Transformer\nd=384, 8 heads'),
                (29.8, 'Multi-Head Output')]:
    box(56, yy, 18, 4.4 if '\n' in txt else 3.6, GREEN_BOX)
    label(65, yy + (2.2 if '\n' in txt else 1.8), txt, size=5.2)
arrow(65, 43.2, 65, 41.2)
arrow(65, 36.6, 65, 33.6)

# operation conditioning arrow
arrow(36.2, 21.6, 55.8, 28.5, cs='arc3,rad=-0.15')
label(45.5, 23.6, r'Operation Conditioning: $\hat{e}_c$', size=4.8, style='italic')

# G-code input (teacher forcing)
label(65, 22.6, 'G-code Input\n(teacher forcing)', size=4.8, style='italic')
arrow(65, 24.6, 65, 25.8)

# --- prediction heads (right column) ---
heads = ['Type Head\n(4 classes)', 'Command Head\n(~30)', 'Parameter Head\n(~10)',
         'Digit Head\n(sign + 6 digits)', 'Legacy Head\n(712 tokens)']
for i, txt in enumerate(heads):
    yy = 47.5 - i * 6.6
    box(83, yy, 15, 5.2, ORANGE)
    label(90.5, yy + 2.6, txt, size=4.8)
    arrow(74.4, 31.6, 82.8, yy + 2.6, cs='arc3,rad=0.1')

# --- bottom: anomaly scoring ---
box(42, 3.5, 15, 5.2, RED)
label(49.5, 6.7, 'Claimed\nG-code', size=5.0, weight='bold')
label(49.5, 1.9, 'from attacker / system', size=4.2, style='italic')
box(71, 3.5, 15, 5.2, RED)
label(78.5, 6.1, 'Predicted\nG-code', size=5.0, weight='bold')
arrow(57.2, 6.1, 70.8, 6.1, style='<|-|>')
label(64, 7.6, r'Mismatch $\rightarrow$ Anomaly Score', size=4.8, weight='bold', color='#b02020')
label(64, 4.5, r'$S_{\mathrm{mean}},\ S_{\mathrm{max}},\ S_{\mathrm{rank}}$', size=4.6)
arrow(90.5, 12.0, 79.5, 8.9, cs='arc3,rad=0.15')

fig.savefig(OUT_DIR / 'architecture_overview.pdf', bbox_inches='tight')
print(f"Saved: {OUT_DIR / 'architecture_overview.pdf'}")

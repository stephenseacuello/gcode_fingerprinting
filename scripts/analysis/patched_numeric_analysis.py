#!/usr/bin/env python3
"""Compare patched-target retrain vs released pre-patch checkpoint on the numeric heads.
(A) headline drift, (B) does slot-5 upturn survive the float-epsilon patch,
(C) entropy-vs-accuracy with patched accuracies, (D) corpus-weighted target change rate.
Writes audit/patched_numeric_analysis.json. Read-only on existing audit files."""
import json, glob
from pathlib import Path
import numpy as np

REPO = Path('/home/seacuello/Documents/gcode_fingerprinting')
DEC = REPO / 'outputs/decoder20260511'
PATCHED = DEC / 'checkpoints/full_window_5fold_patched'
RELEASED = DEC / 'checkpoints/full_window_5fold'
TYPE_NUMERIC = 3
DIGIT_PAD = 10
AXES = {'X': 0, 'Y': 1, 'Z': 2, 'F': 3, 'R': 4}
SLOTW = [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]

def pred_path(root, fold, released):
    if released:
        g = sorted(glob.glob(str(root / f'fold_{fold}' / '*' / 'results' / 'predictions.npz')))
        return Path(g[0]) if g else None
    p = root / f'fold_{fold}' / 'results' / 'predictions.npz'
    return p if p.exists() else None

def compose(sign, digits):
    val = sum(d * w for d, w in zip(digits, SLOTW))
    return -val if sign == 1 else val

def per_slot_acc(z):
    tt = z['type_t']; dp = z['digit_p']; dt = z['digit_t']
    num = (tt == TYPE_NUMERIC)
    corr = np.zeros(6); tot = np.zeros(6)
    for s in range(6):
        m = num & (dt[..., s] != DIGIT_PAD)
        corr[s] = ((dp[..., s] == dt[..., s]) & m).sum(); tot[s] = m.sum()
    return corr, tot

def per_axis_slot_acc(z):
    tt = z['type_t']; pt = z['pt_t']; dp = z['digit_p']; dt = z['digit_t']
    out = {}
    for ax, aid in AXES.items():
        corr = np.zeros(6); tot = np.zeros(6); base = (tt == TYPE_NUMERIC) & (pt == aid)
        for s in range(6):
            m = base & (dt[..., s] != DIGIT_PAD)
            corr[s] = ((dp[..., s] == dt[..., s]) & m).sum(); tot[s] = m.sum()
        out[ax] = (corr, tot)
    return out

def per_axis_mae(z):
    tt = z['type_t']; pt = z['pt_t']; dp = z['digit_p']; dt = z['digit_t']
    sp = z['sign_p']; st = z['sign_t']; out = {}
    for ax, aid in AXES.items():
        m = (tt == TYPE_NUMERIC) & (pt == aid) & (dt[..., 0] != DIGIT_PAD)
        idx = np.argwhere(m); errs = []
        for (i, j) in idx:
            vp = compose(int(sp[i, j]), [int(x) for x in dp[i, j]])
            vt = compose(int(st[i, j]), [int(x) for x in dt[i, j]])
            errs.append(abs(vp - vt))
        out[ax] = (float(np.mean(errs)) if errs else None, len(errs))
    return out

def slot5_by_opclass(zs, fold_npz_root):
    """patched slot-5 accuracy stratified by op-class family."""
    fam_corr = {}; fam_tot = {}
    for f, z in zip(range(1, 6), zs):
        if z is None: continue
        meta = np.load(fold_npz_root / f'fold_{f}/test_sequences.npz', allow_pickle=True)
        names = [str(x) for x in meta['operation_type_names']]
        n = min(len(names), z['type_t'].shape[0])
        tt = z['type_t'][:n]; dp = z['digit_p'][:n]; dt = z['digit_t'][:n]
        num = (tt == TYPE_NUMERIC) & (dt[..., 5] != DIGIT_PAD)
        for i in range(n):
            fam = 'face' if names[i].startswith('face') or names[i] == 'damageface' else \
                  'adaptive' if names[i].startswith('adaptive') or names[i] == 'damageadaptive' else \
                  'pocket' if names[i].startswith('pocket') or names[i] == 'damagepocket' else names[i]
            row = num[i]
            c = ((dp[i][..., 5] == dt[i][..., 5]) & row).sum(); t = row.sum()
            fam_corr[fam] = fam_corr.get(fam, 0) + c; fam_tot[fam] = fam_tot.get(fam, 0) + t
    return {k: round(float(fam_corr[k] / max(fam_tot[k], 1)), 4) for k in fam_corr if fam_tot[k] > 0}

def load_folds(root, released):
    return [np.load(pred_path(root, f, released), allow_pickle=True) if pred_path(root, f, released) else None for f in range(1, 6)]

def agg_per_slot(zs):
    pf = []
    for z in zs:
        if z is None: continue
        c, t = per_slot_acc(z); pf.append(c / np.maximum(t, 1))
    a = np.array(pf); return a.mean(0), a.std(0)

def main():
    patched = load_folds(PATCHED, False); released = load_folds(RELEASED, True)
    report = {}
    pm, ps = agg_per_slot(patched); rm, rs = agg_per_slot(released)
    report['pooled_per_slot_bathtub'] = {
        'slots': ['10^1','10^0','10^-1','10^-2','10^-3','10^-4'],
        'patched_mean': pm.round(4).tolist(), 'patched_std': ps.round(4).tolist(),
        'released_mean': rm.round(4).tolist(), 'released_std': rs.round(4).tolist(),
        'delta_patched_minus_released': (pm - rm).round(4).tolist()}
    axslot = {ax: [] for ax in AXES}
    for z in patched:
        if z is None: continue
        d = per_axis_slot_acc(z)
        for ax in AXES:
            c, t = d[ax]; axslot[ax].append(c / np.maximum(t, 1))
    report['patched_per_axis_slot'] = {ax: np.array(v).mean(0).round(4).tolist() for ax, v in axslot.items() if v}
    report['slot5_by_opclass_family_patched'] = slot5_by_opclass(patched, DEC / 'preprocessed_f98/full_window')
    def mae_folds(zs):
        acc = {ax: [] for ax in AXES}
        for z in zs:
            if z is None: continue
            d = per_axis_mae(z)
            for ax in AXES:
                if d[ax][0] is not None: acc[ax].append(d[ax][0])
        return {ax: (round(float(np.mean(v)), 4) if v else None) for ax, v in acc.items()}
    report['per_axis_mae'] = {'patched': mae_folds(patched), 'released': mae_folds(released),
                              'paper_released_values': {'X': 0.277, 'Y': 0.179, 'Z': 0.042, 'R': 0.018}}
    chg = 0; tot = 0; t0 = 0; tok_match = True
    for zp, zr in zip(patched, released):
        if zp is None or zr is None: continue
        n = min(zp['target_tokens'].shape[0], zr['target_tokens'].shape[0])
        tt = zp['type_t'][:n]; dtp = zp['digit_t'][:n]; dtr = zr['digit_t'][:n]
        if not np.array_equal(zp['target_tokens'][:n], zr['target_tokens'][:n]): tok_match = False
        num = (tt == TYPE_NUMERIC) & (dtp[..., 0] != DIGIT_PAD)
        chg += (num & np.any(dtp != dtr, axis=-1)).sum(); tot += num.sum()
        t0 += (num & (dtp[..., 5] == 0) & (dtr[..., 5] == 9)).sum()
    report['corpus_target_change_rate'] = {
        'target_tokens_identical': bool(tok_match), 'numeric_positions': int(tot),
        'positions_with_any_digit_changed': int(chg), 'fraction_changed': round(float(chg / max(tot, 1)), 4),
        'terminal_zero_9to0_fixed': int(t0), 'frac_terminal_zero_fixed': round(float(t0 / max(tot, 1)), 4)}
    # (C) CORRECTED source entropy from patched (round-encoded) digit_t, paired with patched accuracy.
    def shannon(vals):
        vals = vals[vals != DIGIT_PAD]
        if len(vals) == 0: return None
        _, c = np.unique(vals, return_counts=True); p = c / c.sum()
        return float(-(p * np.log2(p)).sum())
    # pool digit_t across folds per (axis, slot) and pooled-over-axes per slot
    ax_slot_vals = {ax: [[] for _ in range(6)] for ax in AXES}
    pooled_vals = [[] for _ in range(6)]
    for z in patched:
        if z is None: continue
        tt = z['type_t']; pt = z['pt_t']; dt = z['digit_t']; num = (tt == TYPE_NUMERIC)
        for s in range(6):
            pooled_vals[s].append(dt[..., s][num])
            for ax, aid in AXES.items():
                ax_slot_vals[ax][s].append(dt[..., s][num & (pt == aid)])
    report['corrected_pooled_slot_entropy'] = [shannon(np.concatenate(pooled_vals[s])) for s in range(6)]
    ent_axslot = {ax: [shannon(np.concatenate(ax_slot_vals[ax][s])) for s in range(6)] for ax in AXES}
    report['corrected_per_axis_slot_entropy'] = {ax: [round(e, 3) if e is not None else None for e in v] for ax, v in ent_axslot.items()}
    # correlations: pooled n=6, and per-(axis,slot) cells for X/Y/Z/R/F
    def corr(xs, ys):
        if len(xs) < 4: return None
        xs = np.array(xs); ys = np.array(ys); pear = float(np.corrcoef(xs, ys)[0, 1])
        try:
            from scipy import stats; sp = float(stats.spearmanr(xs, ys).statistic)
        except Exception: sp = None
        return {'n': len(xs), 'pearson_r': round(pear, 3), 'spearman_rho': round(sp, 3) if sp is not None else None}
    pe = report['corrected_pooled_slot_entropy']; pa_acc = (np.array([report['pooled_per_slot_bathtub']['patched_mean']])).ravel()
    xs6 = [e for e in pe if e is not None]; ys6 = [pa_acc[s] for s in range(6) if pe[s] is not None]
    report['corr_pooled_slot_corrected'] = corr(xs6, ys6)
    xc, yc = [], []
    for ax in AXES:
        for s in range(6):
            e = ent_axslot[ax][s]; a = report['patched_per_axis_slot'][ax][s]
            if e is not None: xc.append(e); yc.append(a)
    report['corr_per_axis_slot_corrected'] = corr(xc, yc)
    out = DEC / 'audit/patched_numeric_analysis.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)); print('\nWROTE', out)

if __name__ == '__main__':
    main()

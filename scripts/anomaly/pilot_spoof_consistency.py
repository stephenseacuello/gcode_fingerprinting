#!/usr/bin/env python3
"""Paper-6 in-silico PILOT 2: cross-channel / cross-board sensor-spoofing consistency detector.

A PCA reconstruction detector trained on NORMAL multi-channel sensor frames flags physically
implausible streams: spoofing a subset of channels inconsistently pushes a frame off the
normal manifold -> high reconstruction residual. We simulate spoofing attacks on existing
aligned CSVs and sweep the number of spoofed channels to expose the evasion limit (an attacker
who coherently spoofs MORE channels is harder to catch).

HONESTY: this bounds detection of OUR perturbation operators, not arbitrary spoofs. File-level
train/test split (no leakage). All data is existing recordings (no lab collection needed).
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "physical_attack_paper_20260617"

RNG = np.random.RandomState(42)

def sensor_channels(df):
    """Numeric IMU-motion + electrical channels; exclude machine-state/metadata."""
    state = {'t_console','stat','line','posx','posy','posz','mpox','mpoy','mpoz','vel','feed',
             'unit','dist','plane','coor','momo','raw_json'}
    cols = []
    for c in df.columns:
        if c in state: continue
        if df[c].dtype == object: continue
        # keep IMU motion (.Ax/.Gx/.Mx...) and electrical
        if any(k in c for k in ['.A','.G','.M']) or c in ('spindle','x_motor','y_motor','z_motor',
                                                          'spindle_A','x_motor_A','y_motor_A','z_motor_A'):
            cols.append(c)
    return cols

def load_runs(globs):
    files = []
    for g in globs: files += sorted(glob.glob(str(ROOT/'data'/g)))
    return files

def board_of(ch):
    """Group channels by IMU board / electrical bank for the cross-board sweep."""
    return ch.split('.')[0] if '.' in ch else 'electrical'

# ---- spoofing operators (applied to a chosen set of channels on a test frame matrix X[T,C]) ----
def spoof(X, cols_idx, kind, donor=None):
    Y = X.copy()
    for j in cols_idx:
        col = Y[:, j]
        if kind == 'replay' and donor is not None:
            d = donor[:, j]; Y[:, j] = d[:len(col)] if len(d) >= len(col) else np.resize(d, len(col))
        elif kind == 'const':
            Y[:, j] = np.nanmedian(col)
        elif kind == 'scale':
            Y[:, j] = col * 1.5
        elif kind == 'bias':
            Y[:, j] = col + 2.0 * (np.nanstd(col) + 1e-9)
        elif kind == 'dropout':
            Y[:, j] = 0.0
        elif kind == 'timeshift':
            Y[:, j] = np.roll(col, 7)
    return Y

def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]; s = np.r_[neg, pos]
    try: return float(roc_auc_score(y, s))
    except ValueError: return 0.5

def main():
    # use the three active-cut classes (richest dynamics) for a realistic normal manifold
    files = load_runs(['adaptive150025_0*_aligned.csv','face150025_0*_aligned.csv','pocket150025_0*_aligned.csv'])
    # reference schema from first file
    ref = pd.read_csv(files[0], nrows=5)
    chans = sensor_channels(ref)
    boards = sorted(set(board_of(c) for c in chans))
    # keep only channels present in ALL files (robust intersection)
    chans = [c for c in chans if all(c in pd.read_csv(f, nrows=1).columns for f in files)]
    # load all runs; impute per-column NaN with column median (don't drop whole rows)
    runs = []
    for f in files:
        df = pd.read_csv(f)
        X = df[chans].apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.median(numeric_only=True)).to_numpy()
        X = X[~np.isnan(X).any(axis=1)]
        if len(X) >= 60: runs.append((Path(f).name, X))
    name2idx = {c:i for i,c in enumerate(chans)}
    boards = sorted(set(board_of(c) for c in chans))
    # file-level split: 70% train normal, 30% test
    idx = RNG.permutation(len(runs)); ntr = int(0.7*len(runs))
    train = [runs[i] for i in idx[:ntr]]; test = [runs[i] for i in idx[ntr:]]
    Xtr = np.vstack([X for _,X in train])
    mu, sd = Xtr.mean(0), Xtr.std(0)+1e-9
    Ztr = (Xtr-mu)/sd
    # PCA consistency model (keep 95% var)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95, svd_solver='full').fit(Ztr)
    def recon_err(Z):
        return ((Z - pca.inverse_transform(pca.transform(Z)))**2).mean(1)  # per-row MSE
    # per-FRAME (per-row) scoring across all test runs -> large n
    def frame_scores(X):
        return recon_err((X-mu)/sd)
    normal_scores = np.concatenate([frame_scores(X) for _,X in test])
    donor_pool = [X for _,X in train]
    C = len(chans)
    results = {'n_channels': C, 'n_boards': len(boards), 'boards': boards,
               'n_train_runs': len(train), 'n_test_runs': len(test),
               'n_test_frames': int(len(normal_scores)),
               'pca_components': int(pca.n_components_), 'effective_hz_note': '~3.74 Hz',
               'sweeps': {}}
    spoof_kinds = ['replay','const','scale','bias','dropout','timeshift']
    # channel-count sweep: 1 channel, 1 board (~8 ch), half, all
    imu_board = next(b for b in boards if b != 'electrical')
    sweep_sets = {'1_channel':[chans[RNG.randint(C)]],
                  '1_board':[c for c in chans if board_of(c)==imu_board],
                  'half_channels':list(RNG.choice(chans, C//2, replace=False)),
                  'all_channels':chans}
    for sweep_name, cset in sweep_sets.items():
        cidx = [name2idx[c] for c in cset]
        results['sweeps'][sweep_name] = {'n_spoofed': len(cidx)}
        for kind in spoof_kinds:
            spoof_scores = []
            for k,(_,X) in enumerate(test):
                donor = donor_pool[k % len(donor_pool)] if kind=='replay' else None
                Xs = spoof(X, cidx, kind, donor)
                spoof_scores.append(frame_scores(Xs))
            results['sweeps'][sweep_name][kind] = round(auroc(normal_scores, np.concatenate(spoof_scores)),3)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'pilot2_spoof_consistency.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()

"""Confound-free experiment: normal vs damage detection (same workspace).

For each toolpath type, the damage runs traverse the SAME workspace/program as
the normal active runs (verified: identical posx/posy range), so workspace
position is held constant and cannot separate the normal/damage label. Any
discrimination must come from process dynamics (the damage), not position --
directly neutralizing the magnetometer position-confound that dominates the
3-class task, and serving the manufacturing-security (tamper/damage) motivation.

Pooled task: 60 normal-active (adaptive/face/pocket _150025) vs 15 damage
(damage{type}); raw vs air-cut-referenced (subtract per-type air baseline).
"""
import sys, json, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import warnings; warnings.filterwarnings("ignore")

DATA = cfg.DATA_DIR  # data_clean: the vetted set (51 active 17/15/19, 14 damage 5/4/5, 55 air) used by the rest of the paper
TYPES = ["adaptive", "face", "pocket"]

def featurize(path):
    df = utils.load_run(Path(path))
    bins = utils.temporal_bin(df, n_bins=cfg.N_BINS)
    return utils.aggregate_bins_to_run(bins, method="summary")

def collect():
    rows = []  # (vec, label, type)
    mu_air = {}
    for t in TYPES:
        air = [featurize(f) for f in sorted(glob.glob(str(DATA / f"{t}_*_aligned.csv")))]
        mu_air[t] = np.nanmean(np.array(air), axis=0)
        for f in sorted(glob.glob(str(DATA / f"{t}150025_*_aligned.csv"))):
            rows.append((featurize(f), 0, t))   # normal active
        for f in sorted(glob.glob(str(DATA / f"damage{t}_*_aligned.csv"))):
            rows.append((featurize(f), 1, t))   # damage
    X = np.nan_to_num(np.array([r[0] for r in rows], dtype=np.float64))
    y = np.array([r[1] for r in rows])
    mu = np.nan_to_num(np.array([mu_air[r[2]] for r in rows], dtype=np.float64))
    types = np.array([r[2] for r in rows])
    return X, y, mu, types

def rcv(X, y, reps=20):
    accs, f1s, baccs = [], [], []
    for rep in range(reps):
        skf = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE + rep)
        for tr, te in skf.split(X, y):
            s = StandardScaler()
            Xtr = np.nan_to_num(s.fit_transform(X[tr])); Xte = np.nan_to_num(s.transform(X[te]))
            c = RandomForestClassifier(**cfg.RF_PARAMS); c.fit(Xtr, y[tr]); p = c.predict(Xte)
            accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, zero_division=0))
            baccs.append(balanced_accuracy_score(y[te], p))
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "bal_acc": float(np.mean(baccs)), "n_eval": len(accs)}

def channel_importance(X, y):
    """Sum RF Gini importance by channel type (position-confound check)."""
    names = utils.make_stat_column_names(cfg.SENSOR_COLS)  # 770
    feat_names = [f"m_{n}" for n in names] + [f"s_{n}" for n in names]  # 1540
    s = StandardScaler(); Xs = np.nan_to_num(s.fit_transform(X))
    c = RandomForestClassifier(**cfg.RF_PARAMS); c.fit(Xs, y)
    imp = c.feature_importances_
    def chtype(fn):
        for ch, key in [("magnetometer", ".M"), ("accelerometer", ".A"), ("gyroscope", ".G"),
                        ("acoustic_RMS", ".RMS"), ("temperature", ".Temperature"),
                        ("pressure", ".Pressure"), ("proximity", ".Proximity"), ("color", "Color")]:
            if key in fn: return ch
        if any(e in fn for e in cfg.ELECTRICAL_COLS) and not any(m in fn for m in cfg.ARDUINO_MODULES):
            return "electrical"
        return "other"
    agg = {}
    for fn, w in zip(feat_names, imp):
        agg[chtype(fn)] = agg.get(chtype(fn), 0.0) + float(w)
    return dict(sorted(agg.items(), key=lambda kv: -kv[1]))

if __name__ == "__main__":
    X, y, mu, types = collect()
    print(f"n={len(y)}  normal={int((y==0).sum())} damage={int((y==1).sum())}  "
          f"majority-class acc={max((y==0).mean(),(y==1).mean()):.3f}  features={X.shape[1]}")
    Xsub = X - mu
    out = {"n": len(y), "n_normal": int((y==0).sum()), "n_damage": int((y==1).sum()),
           "majority_acc": float(max((y==0).mean(), (y==1).mean())), "n_features": X.shape[1]}
    for name, Xx in [("raw", X), ("subtracted", Xsub)]:
        out[name] = rcv(Xx, y)
        r = out[name]
        print(f"  {name:11s}: acc={r['acc']:.3f}±{r['acc_std']:.3f}  F1(damage)={r['f1']:.3f}  bal_acc={r['bal_acc']:.3f}")
    # per-type (within-type: position truly constant)
    out["per_type"] = {}
    for t in TYPES:
        m = types == t
        if (y[m] == 1).sum() >= 2 and (y[m] == 0).sum() >= 2:
            out["per_type"][t] = {"raw": rcv(X[m], y[m], reps=20), "subtracted": rcv(Xsub[m], y[m], reps=20),
                                  "n_normal": int((y[m]==0).sum()), "n_damage": int((y[m]==1).sum())}
            print(f"  [{t}] raw acc={out['per_type'][t]['raw']['acc']:.3f} / sub acc={out['per_type'][t]['subtracted']['acc']:.3f} "
                  f"(F1 raw={out['per_type'][t]['raw']['f1']:.3f} sub={out['per_type'][t]['subtracted']['f1']:.3f})")
    out["channel_importance_subtracted"] = channel_importance(Xsub, y)
    print("  channel importance (subtracted):", {k: round(v, 3) for k, v in out["channel_importance_subtracted"].items()})
    outpath = cfg.RESULTS_DIR / "experiment_damage_detection.json"
    json.dump(out, open(outpath, "w"), indent=2)
    print(f"\nSaved {outpath}")

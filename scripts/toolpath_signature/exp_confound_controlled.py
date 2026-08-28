"""Backing artifact for the progressive channel-exclusion table (tab:confound_controlled).
Regenerated with the corrected electrical filter (electrical = 8 true channels = 112
features, NOT the 350 produced by the 'spindle'->'spindle2' substring leak)."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolpath_signature import config as cfg, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings("ignore")

df = utils.load_parquet(cfg.RUN_VECTORS_DIR / "raw_temporal_summary.parquet")
meta = ["toolpath", "run_id", "prefix"]
cols = [c for c in df.columns if c not in meta]
y = LabelEncoder().fit_transform(df["toolpath"].values)

def has(c, keys): return any(k in c for k in keys)
def is_elec(c): return any(e in c for e in cfg.ELECTRICAL_COLS) and not any(m in c for m in cfg.ARDUINO_MODULES)

configs = {
    "all": cols,
    "no_magnetometers": [c for c in cols if not has(c, [".Mx", ".My", ".Mz"])],
    "no_mag_color_pressure_proximity": [c for c in cols if not has(c, [".Mx",".My",".Mz","Color",".Pressure",".Proximity"])],
    "accel_gyro_rms_temp": [c for c in cols if has(c, [".Ax",".Ay",".Az",".Gx",".Gy",".Gz",".RMS",".Temperature"])],
    "accel_gyro": [c for c in cols if has(c, [".Ax",".Ay",".Az",".Gx",".Gy",".Gz"])],
    "accel_only": [c for c in cols if has(c, [".Ax",".Ay",".Az"])],
    "electrical_only": [c for c in cols if is_elec(c)],
}

def cv(Xc):
    accs = []
    for rep in range(20):
        skf = StratifiedKFold(5, shuffle=True, random_state=cfg.RANDOM_STATE + rep)
        for tr, te in skf.split(Xc, y):
            s = StandardScaler(); Xtr = np.nan_to_num(s.fit_transform(Xc[tr])); Xte = np.nan_to_num(s.transform(Xc[te]))
            c = RandomForestClassifier(**cfg.RF_PARAMS); c.fit(Xtr, y[tr])
            accs.append(accuracy_score(y[te], c.predict(Xte)))
    return float(np.mean(accs)), float(np.std(accs))

out = {}
for name, cc in configs.items():
    X = np.nan_to_num(df[cc].values.astype(np.float64))
    m, s = cv(X)
    out[name] = {"acc_mean": m, "acc_std": s, "n_features": len(cc)}
    print(f"  {name:34s}: {m*100:.1f} ± {s*100:.1f}  [{len(cc)} feat]")
json.dump(out, open(cfg.RESULTS_DIR / "experiment_confound_controlled.json", "w"), indent=2)
print("saved experiment_confound_controlled.json")

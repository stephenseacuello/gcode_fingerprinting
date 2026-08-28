"""Class-conditional modal command predictor accuracy, 5-fold (canonical).
Per operation class, the modal TRUE command over command-positions; accuracy =
fraction of test command-positions whose true command equals that class's modal command.
Weighted across the 5 folds. Stricter null used in the threat-model/confound discussion."""
import json, glob, numpy as np
from pathlib import Path
DEC = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260511')
REL = DEC/'checkpoints/full_window_5fold'
TYPE_COMMAND = 1
per_fold=[]
for f in range(1,6):
    g=sorted(glob.glob(str(REL/f'fold_{f}'/'*'/'results'/'predictions.npz')))
    if not g: continue
    z=np.load(g[0], allow_pickle=True)
    meta=np.load(DEC/f'preprocessed_f98/full_window/fold_{f}/test_sequences.npz', allow_pickle=True)
    names=[str(x) for x in meta['operation_type_names']]
    tt=z['type_t']; ct=z['cmd_t']; n=min(len(names), tt.shape[0])
    by_cls={}
    for i in range(n):
        for c in ct[i][tt[i]==TYPE_COMMAND]:
            by_cls.setdefault(names[i], []).append(int(c))
    corr=tot=0
    for cls,cmds in by_cls.items():
        if not cmds: continue
        vals,cnts=np.unique(cmds, return_counts=True); modal=vals[cnts.argmax()]
        corr+=int((np.array(cmds)==modal).sum()); tot+=len(cmds)
    per_fold.append(corr/tot)
out={'metric':'class-conditional modal command predictor (test-derived per-op-class modal), accuracy over COMMAND positions',
     'per_fold_accuracy':[round(x,4) for x in per_fold],
     'mean_accuracy':round(float(np.mean(per_fold)),4),
     'std_accuracy':round(float(np.std(per_fold,ddof=1)),4),
     'n_folds':len(per_fold),
     'source':'checkpoints/full_window_5fold/fold_*/results/predictions.npz cmd_t@COMMAND x preprocessed_f98/full_window/fold_*/test_sequences.npz operation_type_names'}
(DEC/'audit/class_conditional_modal_5fold.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

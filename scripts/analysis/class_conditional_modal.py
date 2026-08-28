"""Class-conditional modal command predictor (FIRST-command-position basis), 5-fold, canonical.
Per operation class, the modal command id at the FIRST COMMAND position of each window;
accuracy = fraction of test windows whose first command equals their class's modal first-command.
Computed from cmd_t at the first type==COMMAND position (real command ids 0-5; avoids the
original fold-1 artifact of UNK appearing as a 'modal command'). Stricter null for the
threat-model/confound discussion."""
import json, glob, numpy as np
from pathlib import Path
DEC = Path('/home/seacuello/Documents/gcode_fingerprinting/outputs/decoder20260511')
REL = DEC/'checkpoints/full_window_5fold'; TYPE_COMMAND=1
per_fold=[]
for f in range(1,6):
    g=sorted(glob.glob(str(REL/f'fold_{f}'/'*'/'results'/'predictions.npz')))
    if not g: continue
    z=np.load(g[0], allow_pickle=True)
    meta=np.load(DEC/f'preprocessed_f98/full_window/fold_{f}/test_sequences.npz', allow_pickle=True)
    names=[str(x) for x in meta['operation_type_names']]
    tt=z['type_t']; ct=z['cmd_t']; n=min(len(names), tt.shape[0])
    firsts={}  # op-class -> list of first-command ids
    for i in range(n):
        pos=np.where(tt[i]==TYPE_COMMAND)[0]
        if len(pos)==0: continue
        firsts.setdefault(names[i], []).append(int(ct[i][pos[0]]))
    corr=tot=0
    for cls,cmds in firsts.items():
        vals,cnts=np.unique(cmds, return_counts=True); modal=vals[cnts.argmax()]
        corr+=int((np.array(cmds)==modal).sum()); tot+=len(cmds)
    per_fold.append(corr/tot)
out={'metric':'class-conditional modal command predictor, FIRST-command-position basis, accuracy over test windows',
     'per_fold_accuracy':[round(x,4) for x in per_fold],
     'mean_accuracy':round(float(np.mean(per_fold)),4),
     'std_accuracy':round(float(np.std(per_fold,ddof=1)),4),'n_folds':len(per_fold),
     'note':'Computed from cmd_t at first COMMAND position (ids 0-5), avoiding the prior fold-1 UNK-as-modal artifact.',
     'source':'checkpoints/full_window_5fold/fold_*/results/predictions.npz x preprocessed_f98/full_window/fold_*/test_sequences.npz'}
(DEC/'audit/class_conditional_modal.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))

import re, glob, math, os
from collections import Counter
MAX_INT, MAX_DEC = 2, 4
PREC = {'X':0.001,'Y':0.001,'Z':0.001,'R':0.0001,'F':1.0}
PARAM_RE = re.compile(r'([XYZRF])(-?\d+\.?\d*)')
def enc(val, precision=None):
    a=abs(float(val))
    if precision: a=round(a/precision)*precision
    tot=MAX_INT+MAX_DEC; sc=10**MAX_DEC
    s=min(int(round(a*sc)),10**tot-1)
    return [(s//(10**(tot-1-p)))%10 for p in range(tot)]
def H(c):
    t=sum(c.values())
    if t==0: return 0.0
    return -sum((n/t)*math.log2(n/t) for n in c.values() if n)
def fam(fn):
    b=os.path.basename(fn).replace('.gcode','')
    for f in ('face','adaptive','pocket'):
        if b.startswith(f): return f
    return b
# collect per (axis): list of values; per (family,axis): values
vals={ax:[] for ax in PREC}; famvals={}
for fn in glob.glob('data_clean/*.gcode'):
    fm=fam(fn)
    for line in open(fn):
        line=re.sub(r'\(.*?\)','',line); line=line.split(';')[0]
        for ax,v in PARAM_RE.findall(line):
            try: x=float(v)
            except: continue
            vals[ax].append(x); famvals.setdefault((fm,ax),[]).append(x)
SL=['10^1','10^0','10^-1(tenths)','10^-2(hund)','10^-3(thou)','10^-4(tenthou)']
print("=== PER-AXIS per-slot entropy: SOURCE(4-dec raw) vs TOKENIZED(axis precision) ===")
for ax in ['X','Y','Z','R','F']:
    if not vals[ax]: continue
    src=[Counter() for _ in range(6)]; tok=[Counter() for _ in range(6)]
    for x in vals[ax]:
        ds=enc(x); dt=enc(x,PREC[ax])
        for s in range(6): src[s][ds[s]]+=1; tok[s][dt[s]]+=1
    print(f"\n{ax} (precision={PREC[ax]}, n={len(vals[ax])}):")
    print("  slot           SOURCE  TOKENIZED")
    for s in range(6):
        print(f"  {SL[s]:>14} {H(src[s]):6.3f}   {H(tok[s]):6.3f}")
print()
print("=== THREE REGIMES at thousandths (slot-4, 10^-3) X-axis: SOURCE vs TOKENIZED ===")
for fm in ['face','adaptive','pocket']:
    xs=famvals.get((fm,'X'),[])
    if not xs: continue
    cs=Counter(enc(x)[4] for x in xs); ct=Counter(enc(x,0.001)[4] for x in xs)
    print(f"  {fm:>9}: source H={H(cs):.3f}  tokenized H={H(ct):.3f}  (n={len(xs)})")

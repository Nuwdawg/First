import sys,time; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
t0=time.time()
C=load_val_cands('full'); G=GT(list(C)); print('loaded',time.time()-t0,flush=True)
print('floor  nPred | entry PQ (tp fp fn) | pooled PQ sq (tp fp fn)')
for floor in [0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0.02]:
    ent=[];pool=[];n=0
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,floor); e,p=score_photo(cv,ar,G.g[ph]); ent+=e; pool.append(p); n+=len(ar)
    e,p=pq_from(ent),pq_from(pool)
    print('%.2f %5d | %.4f (%d %d %d) | %.4f %.3f (%d %d %d)'%(floor,n,e['pq'],e['tp'],e['fp'],e['fn'],p['pq'],p['sq'],p['tp'],p['fp'],p['fn']),flush=True)
print('time',time.time()-t0)

import sys,time; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); G=GT(list(C))
def pqc(c):
    s,tp,fp,fn=c; den=tp+0.5*fp+0.5*fn
    return s/den if den else None
def run(floor):
    ent={};pool={}
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,floor); e,p=score_photo(cv,ar,G.g[ph]); ent[ph]=e; pool[ph]=p
    allrec=[c for ph in ent for c in ent[ph]]
    v1=pq_from(allrec)['pq']
    rec=[pqc(c) for c in allrec]; rec_valid=[r for r in rec if r is not None]
    v2=np.mean(rec_valid)
    # per photo: pool records' counts of that photo, then mean over photos
    v3=np.mean([pq_from(ent[ph])['pq'] for ph in ent])
    v4=np.mean([pqc(pool[ph]) or 0 for ph in pool])
    v5=pq_from(list(pool.values()))['pq']
    # first-record-only (one annotator per photo: the first listed)
    v6=pq_from([ent[ph][0] for ph in ent])['pq']
    return v1,v2,v3,v4,v5,v6,len(rec)-len(rec_valid)
print('floor | V1 global entry | V2 mean per-record | V3 mean per-photo | V4 mean per-photo pooled | V5 global pooled | V6 first record only')
for f in [0.9,0.85,0.8,0.75,0.7,0.6]:
    r=run(f); print('%.2f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | empty-den records %d'%((f,)+r))

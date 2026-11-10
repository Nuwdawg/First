import sys,time; sys.path.insert(0,'/home/user/work/an')
from harness import *
S=val_sources(); nd={ph:sum(c.score>=0.85 for c in _C[ph]) for ph in PHOTOS} if False else None
import harness
C0=harness._C
ratio={}
for ph in PHOTOS:
    n=max(sum(c.score>=0.85 for c in C0[ph]),1)
    for e,items in G.g[ph]: ratio[e]=len(items)/n
CL={ph:cluster(S[ph],link=0.5) for ph in PHOTOS}
def table(O,label,floors):
    for floor in floors:
        ent=[];sub=[]
        for ph in PHOTOS:
            cv,ar,k=paint_labels(O[ph],floor); e,p=score_photo(cv,ar,G.g[ph])
            for (en,items),c in zip(G.g[ph],e):
                ent.append(c)
                if ratio[en]<0.95: sub.append(c)
        a=pq_from(ent); s=pq_from(sub)
        print('%s %.2f | all %.4f (%d/%d/%d) | sparse-relative %.4f (%d/%d/%d)'%(label,floor,a['pq'],a['tp'],a['fp'],a['fn'],s['pq'],s['tp'],s['fp'],s['fn']),flush=True)
single={ph:C0[ph] for ph in PHOTOS}
table(single,'single',[0.8,0.85,0.88,0.9,0.92,0.94,0.96])
for mode,thr in [('vote',0.5),('vote',0.6)]:
    O={ph:[consensus(k,8,mode,thr) for k in cl] for ph,cl in CL.items()}
    table(O,'d4-%s%.1f-mean8'%(mode,thr),[0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.94])
    for ph in PHOTOS:
        for o in O[ph]: o.score=o.meta['max']*(o.meta['n']/8)
    table(O,'d4-%s%.1f-max*frac'%(mode,thr),[0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.94])

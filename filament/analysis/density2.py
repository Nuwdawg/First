import sys,time; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); G=GT(list(C))
nd={ph:sum(c.score>=0.85 for c in C[ph]) for ph in C}
ratio={e:len(items)/max(nd[ph],1) for ph in C for e,items in G.g[ph]}
r=np.array(list(ratio.values())); print('ratio GT/pred@0.85 per record: mean %.2f median %.2f; quantiles'%(r.mean(),np.median(r)),np.quantile(r,[.1,.25,.5,.75,.9]).round(2))
def run(floor,min_area=100):
    out={}
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,floor,min_area=min_area); e,p=score_photo(cv,ar,G.g[ph])
        for (ename,items),cc in zip(G.g[ph],e): out[ename]=cc
    return out
bins=[(0,0.7),(0.7,0.95),(0.95,1.25),(1.25,99)]
print('bins',[(b,sum((r>=b[0])&(r<b[1]))) for b in bins])
for floor in [0.7,0.8,0.85,0.9,0.93,0.95,0.97]:
    o=run(floor); line='%.2f |'%floor
    for b in bins:
        a=pq_from([o[e] for e in o if b[0]<=ratio[e]<b[1]]); line+=' %.4f (%d/%d/%d) |'%(a['pq'],a['tp'],a['fp'],a['fn'])
    # reweighted: records weighted so that low-ratio records dominate (emulating GT/pred ratio 0.70)
    print(line,flush=True)
# subset emulating ratio ~0.70: take records with ratio<0.95
sel=[e for e in ratio if ratio[e]<0.95]; print('subset ratio<0.95: n',len(sel),'mean ratio',np.mean([ratio[e] for e in sel]).round(2))

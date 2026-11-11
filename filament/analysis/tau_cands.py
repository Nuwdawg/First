"""Candidate rules scored on validation at several match thresholds (leaderboard may use a stricter one than 0.5)."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
import fast2
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb')),'ctrl31':load('ctrl31','val')}
photos=sorted(M['e40']); G=GT(photos)
def ious(objs,floor):
    L=[]
    for ph in photos:
        cv,ar,k=paint_labels(objs[ph],floor)
        L.append([fast2.iou_mat(cv,ar,items) for e,items in G.g[ph]])
    return L
def pq_tau(L,tau,idx=None):
    s=tp=fp=fn=0
    for i,recs in enumerate(L):
        if idx is not None and i not in idx: continue
        for Mx in recs:
            ng,npr=Mx.shape
            if ng==0: fp+=npr; continue
            if npr==0: fn+=ng; continue
            hit=Mx>tau; s+=Mx[hit].sum(); tp+=hit.sum(); fp+=(hit.sum(0)==0).sum(); fn+=(hit.sum(1)==0).sum()
    return s/(tp+0.5*fp+0.5*fn)
rules={}
S={ph:[c for c in M['e40'][ph] if c.meta['view']==0] for ph in photos}
for f in (0.8,0.85,0.88,0.9): rules['deployed single @%.2f'%f]=ious(S,f)
O1={ph:d4_objects(M['e40'][ph]) for ph in photos}
for agg,fl in [('mean8',(0.8,0.85,0.88)),('maxfrac',(0.88,0.9,0.92,0.94))]:
    for ph in photos:
        for o in O1[ph]: o.score=o.meta[agg]
    for f in fl: rules['e40 8-view %s @%.2f'%(agg,f)]=ious(O1,f)
for subset,tag in [(('e40','swins','swinb'),'three'),(('e40','swins','swinb','ctrl31'),'four')]:
    for agg,fl in [('mean8',(0.8,0.85,0.88,0.9)),('maxfrac',(0.88,0.9,0.92,0.94))]:
        O={ph:multi_objects({m:M[m][ph] for m in subset},agg) for ph in photos}
        for f in fl: rules['%s %s @%.2f'%(tag,agg,f)]=ious(O,f)
pickle.dump(rules,open('/home/user/work/an/tau_rules.pkl','wb'))
taus=(0.5,0.55,0.58,0.6,0.62,0.65)
print('%-28s '%'rule'+' '.join('t=%.2f'%t for t in taus))
for n,L in rules.items(): print('%-28s '%n+' '.join('%.4f'%pq_tau(L,t) for t in taus),flush=True)
# paired bootstrap at tau=0.6: best three/four vs deployed
rng=np.random.default_rng(0)
def boot(a,b,tau,n=1000):
    d=[]
    for _ in range(n):
        idx=rng.integers(0,len(photos),len(photos)); cnt=collections.Counter(idx.tolist())
        def pqw(L):
            s=tp=fp=fn=0
            for i,w in cnt.items():
                for Mx in L[i]:
                    ng,npr=Mx.shape
                    if ng==0: fp+=w*npr; continue
                    if npr==0: fn+=w*ng; continue
                    hit=Mx>tau; s+=w*Mx[hit].sum(); tp+=w*hit.sum(); fp+=w*(hit.sum(0)==0).sum(); fn+=w*(hit.sum(1)==0).sum()
            return s/(tp+0.5*fp+0.5*fn)
        d.append(pqw(rules[a])-pqw(rules[b]))
    d=np.array(d); return d.mean(),np.quantile(d,[.025,.975])
for a in ('three maxfrac @0.92','three mean8 @0.88','four maxfrac @0.90','e40 8-view mean8 @0.85'):
    for tau in (0.5,0.6):
        m,ci=boot(a,'deployed single @0.85',tau); print('%-24s vs deployed at tau %.2f: %+.4f [%+.4f,%+.4f]'%(a,tau,m,ci[0],ci[1]),flush=True)

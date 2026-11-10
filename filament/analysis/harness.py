"""Evaluation harness: per-photo pooled/entry components, paired bootstrap over photographs."""
import sys,time,pickle,os; sys.path.insert(0,'/home/user/work/an')
from cluster import *
_C=load_val_cands('full'); _D=load_val_cands('d4'); G=GT(list(_C))
for ph in _C:
    for c in _C[ph]: c.meta['src']=0
    for c in _D[ph]: c.meta['src']=int(c.meta['view'])
PHOTOS=list(_C)
def val_sources(): return {ph:_C[ph]+_D[ph] for ph in PHOTOS}
def evaluate(objs_by_photo,floor,min_area=100):
    """returns dict photo-> (entry counts list, pooled counts, n)"""
    out={}
    for ph in PHOTOS:
        cv,ar,k=paint_labels(objs_by_photo[ph],floor,min_area=min_area); e,p=score_photo(cv,ar,G.g[ph]); out[ph]=(e,p,len(ar))
    return out
def summarize(res):
    ent=[c for ph in PHOTOS for c in res[ph][0]]; pool=[res[ph][1] for ph in PHOTOS]
    return pq_from(ent),pq_from(pool),sum(res[ph][2] for ph in PHOTOS)
def pooled_pq(res,idx=None):
    ph=PHOTOS if idx is None else [PHOTOS[i] for i in idx]
    s=sum(res[p][1][0] for p in ph); tp=sum(res[p][1][1] for p in ph); fp=sum(res[p][1][2] for p in ph); fn=sum(res[p][1][3] for p in ph)
    return s/(tp+0.5*fp+0.5*fn)
def paired_boot(resA,resB,n=2000,seed=0):
    rng=np.random.default_rng(seed); d=[]
    A=np.array([resA[p][1] for p in PHOTOS],float); B=np.array([resB[p][1] for p in PHOTOS],float)
    def pq(M): return M[:,0].sum()/(M[:,1].sum()+0.5*M[:,2].sum()+0.5*M[:,3].sum())
    for _ in range(n):
        i=rng.integers(0,len(PHOTOS),len(PHOTOS)); d.append(pq(A[i])-pq(B[i]))
    d=np.array(d); return pq(A)-pq(B),np.quantile(d,[0.025,0.975]),(d>0).mean()

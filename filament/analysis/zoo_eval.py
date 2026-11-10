"""Standalone and consensus evaluation of zoo models on validation (per-annotator PQ)."""
import sys,time,pickle,itertools; sys.path.insert(0,'/home/user/work/an')
from zoo import *
from cluster import *
from fast2 import GT, paint_labels, score_photo
import fast2
def boot(resA,resB,photos,n=1000,seed=0):
    """paired bootstrap over photographs of per-annotator PQ difference; res: photo->list of record counts"""
    rng=np.random.default_rng(seed)
    def tot(res,ph): 
        a=np.array([c for c in res[ph]],float); return a.sum(0)
    A=np.array([tot(resA,p) for p in photos]); B=np.array([tot(resB,p) for p in photos])
    pq=lambda M: M[:,0].sum()/(M[:,1].sum()+0.5*M[:,2].sum()+0.5*M[:,3].sum())
    d=[]
    for _ in range(n):
        i=rng.integers(0,len(photos),len(photos)); d.append(pq(A[i])-pq(B[i]))
    d=np.array(d); return pq(A)-pq(B),np.quantile(d,[0.025,0.975]),(d>0).mean()
def run(objs,G,photos,floor,min_area=100):
    res={}
    for ph in photos:
        cv,ar,k=paint_labels(objs[ph],floor,min_area=min_area); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e
    return res
def pq_of(res): return pq_from([c for ph in res for c in res[ph]])
if __name__=='__main__':
    names=sys.argv[1].split(',') if len(sys.argv)>1 else list(ZOO)
    V={n:load(n,'val') for n in names}
    photos=sorted(V[names[0]]); G=GT(photos)
    FL=[0.7,0.75,0.8,0.85,0.88,0.9,0.92]
    base=None
    print('== standalone single view')
    for n in names:
        O={ph:[c for c in V[n][ph] if c.meta['view']==0] for ph in photos}
        best=max(((pq_of(run(O,G,photos,f))['pq'],f) for f in FL))
        r=run(O,G,photos,0.85); a=pq_of(r)
        if n=='e40': base=r
        print('%-8s @0.85 %.4f (%d/%d/%d) | best %.4f @%.2f'%(n,a['pq'],a['tp'],a['fp'],a['fn'],best[0],best[1]),flush=True)
    pickle.dump(base,open('/home/user/work/an/base_res.pkl','wb'))

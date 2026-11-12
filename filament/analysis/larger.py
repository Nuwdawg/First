"""'Prefer the larger candidate' for the annotators who draw bigger filaments.
After the normal painting (score >= floor), look at the model's other candidates L (lo <= score): if L contains
kept masks (>= cover of their area inside L), is at least `grow` times their total area and does not run over
other kept masks, replace the contained pieces by L. Evaluated by annotator population on validation."""
import sys,pickle,numpy as np,collections,itertools; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast import Cand
from fast2 import GT, score_photo, paint_labels, H, W
from lib import load_train, pq_from
from multiprocessing import Pool
Z='/home/user/work/an/zoo'; TARGET={'0101','0102','0104','0501'}
def full(o):
    m=np.zeros((H,W),bool); m[o.y0:o.y0+o.mask.shape[0],o.x0:o.x0+o.mask.shape[1]]=o.mask; return m
def prefer_larger(pool,floor,lo,grow,cover=0.8,spill=0.1,maxgrow=6.0,min_area=100):
    """pool: candidates with .score; returns final list of Cand-like objects in paint order"""
    cv,ar,kept=paint_labels(pool,floor,min_area=min_area)
    n=len(ar)
    if n==0: return kept,0
    keptset=set(id(k) for k in kept); repl=0
    alive=np.ones(n+1,bool); extra=[]
    for L in sorted([c for c in pool if c.score>=lo and id(c) not in keptset and c.area>=min_area],key=lambda c:-c.score):
        sub=cv[L.y0:L.y0+L.mask.shape[0],L.x0:L.x0+L.mask.shape[1]]
        lab=sub[L.mask]; cnt=np.bincount(lab,minlength=n+1)
        inside=[j for j in range(1,n+1) if alive[j] and cnt[j]>=cover*ar[j-1]]
        if not inside: continue
        tot=sum(ar[j-1] for j in inside)
        if L.area<grow*tot or L.area>maxgrow*tot: continue
        others=sum(cnt[j] for j in range(1,n+1) if alive[j] and j not in inside)
        if others>spill*L.area: continue
        # accept: remove the pieces, paint L where free
        for j in inside: alive[j]=False; cv[cv==j]=0
        free=L.mask&(sub==0)
        if free.sum()<min_area: continue
        n+=1; sub[free]=n; ar=np.append(ar,int(free.sum())); alive=np.append(alive,True); repl+=1
    out=[]
    for j in range(1,n+1):
        if not alive[j]: continue
        m=(cv==j); ys,xs=np.nonzero(m)
        if len(ys)==0: continue
        c=object.__new__(Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,m[y0:y1,x0:x1].copy(),int(m.sum()),1.0,{}; out.append(c)
    return out,repl
if __name__=='__main__':
    M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
    photos=sorted(M['e40']); G=GT(photos); d,byimg,PH=load_train()
    POOL={'single':{ph:[c for c in M['e40'][ph] if c.meta['view']==0] for ph in photos}}
    def mk(ph):
        o=multi_objects({m:M[m][ph] for m in M},'mean8'); return ph,o
    with Pool(4) as p: POOL['three_mean8']=dict(p.map(mk,photos,chunksize=4))
    def ev(args):
        src,floor,lo,grow,cover,spill=args; res={}; R=0; N=0
        for ph in photos:
            if lo is None: kept=paint_labels(POOL[src][ph],floor)[2]; r=0
            else: kept,r=prefer_larger(POOL[src][ph],floor,lo,grow,cover,spill)
            cv,ar,k=paint_labels(kept,-1,min_area=1,order=kept); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e; R+=r; N+=len(ar)
        return args,(res,R,N)
    jobs=[]
    for src,floor in (('single',0.85),('three_mean8',0.88)):
        jobs.append((src,floor,None,0,0,0))
        for lo in (0.3,0.5,0.7):
            for grow in (1.2,1.5):
                for cover,spill in ((0.8,0.1),(0.6,0.2)): jobs.append((src,floor,lo,grow,cover,spill))
    with Pool(4) as p: R=dict(p.map(ev,jobs,chunksize=1))
    pickle.dump(R,open(Z+'/larger.pkl','wb'))
    sels={'all':lambda ph,e:True,'target':lambda ph,e:e[:4] in TARGET,'k=3':lambda ph,e:len(PH[ph])==3,'other':lambda ph,e:e[:4] not in TARGET}
    print('source      floor  lo   grow cover spill | swaps masks | '+'  '.join('%7s'%s for s in sels))
    for j in jobs:
        res,r,n=R[j]
        print('%-11s %.2f  %-4s %.1f  %.1f   %.1f  | %4d  %4d  | '%(j[0],j[1],j[2],j[3],j[4],j[5],r,n)+'  '.join('%.4f '%pq_from([c for ph in photos for e,c in zip(PH[ph],res[ph]) if f(ph,e)])['pq'] for f in sels.values()),flush=True)

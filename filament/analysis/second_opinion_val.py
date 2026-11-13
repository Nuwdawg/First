"""Validation analogue of the 'second opinion' step: a vote of eight files from the e40 recipe family, and five
files built only from the Swin-S and Swin-B hold-out models as an independent second opinion. Do vote objects that
no second-opinion file contains hurt, and do unanimous second-opinion objects outside the vote help?"""
import sys,pickle,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
from decode import *
from fast import Cand
from fast2 import GT, score_photo, paint_labels
from lib import load_train, pq_from
import cluster as CLU
import tickets13 as T
from multiprocessing import Pool
M=T.sources('val'); photos=sorted(M['e40']); G=GT(photos)
def d4file(cands,floor):
    objs=d4_objects(cands)
    for o in objs: o.score=o.meta['mean8']
    return T.as_file.__wrapped__(objs,floor) if hasattr(T.as_file,'__wrapped__') else paintfile(objs,floor)
def paintfile(objs,floor):
    cv,ar,kept=paint_labels(objs,floor); L=[]
    for i in range(len(ar)):
        m=(cv==i+1); ys,xs=np.nonzero(m); c=object.__new__(Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,m[y0:y1,x0:x1].copy(),int(ar[i]),1.0,{}; L.append(c)
    return L
def photo(ph):
    per={n:M[n].get(ph,[]) for n in M}
    sv=lambda n,f: paintfile([c for c in per[n] if c.meta.get('view',0)==0],f)
    A={'e40':sv('e40',0.85),'e40d4':d4file(per['e40'],0.85),'ctrl31d4':d4file(per['ctrl31'],0.85),'ctrl31':sv('ctrl31',0.88),'ctrl7':sv('ctrl7',0.90),'focal':sv('focal',0.90),'head':sv('head',0.90),'ftA6':sv('ftA6',0.85)}
    pool=[]
    for n in ('swins','swinb'):
        for o in d4_objects(per[n]):
            o.score=o.meta['mean8']; o.meta['src']=n; pool.append(o)
    two=[consensus(k,2,'vote',0.5) for k in cluster(pool,link=0.5,min_score=0.01)]
    for o in two: o.score=o.meta['sum']/2
    B={'swins':sv('swins',0.90),'swinb':sv('swinb',0.90),'swinsd4':d4file(per['swins'],0.88),'swinbd4':d4file(per['swinb'],0.88),'two85':paintfile(two,0.85)}
    cands=[]
    for n,L in A.items():
        for c in L: c.meta['src']=n; cands.append(c)
    objs=[]
    for cl in CLU.cluster(cands,link=0.5,min_score=0.0):
        o=CLU.consensus(cl,len(A),'vote',0.5); v=o.meta['n']; o.score=v/len(A)
        o.meta={'v':v,'cons':sum(any(CLU.ciou(o,c)>=0.5 for c in B[n]) for n in B)}; objs.append(o)
    rest=[]
    for n,L in B.items():
        for c in L:
            if not any(CLU.ciou(c,o)>=0.5 for o in objs): c.meta['src']=n; rest.append(c)
    new=[]
    for cl in CLU.cluster(rest,link=0.5,min_score=0.0):
        o=CLU.consensus(cl,len(B),'vote',0.5); k=o.meta['n']; o.score=-1.0; o.meta={'v':0,'cons':k}; new.append(o)
    def run(keep,addnew=None):
        sel=[o for o in objs if keep(o)]+([o for o in new if addnew(o)] if addnew else [])
        cv,ar,k=paint_labels(sel,-2,min_area=100); return score_photo(cv,ar,G.g[ph])[0],len(ar)
    base=lambda o:o.meta['v']>=4
    V={'base: at least 4 of 8':run(base)}
    for lim in (5,6,7):
        V['drop vote objects (v<=%d) no second-opinion file has'%lim]=run(lambda o,lim=lim:base(o) and not (o.meta['cons']==0 and o.meta['v']<=lim))
    V['drop vote objects (v<=7) at most one second-opinion file has']=run(lambda o:base(o) and not (o.meta['cons']<=1 and o.meta['v']<=7))
    V['add outside objects all five second-opinion files have']=run(lambda o:base(o) or o.meta['cons']==5)
    V['add outside objects at least four have']=run(lambda o:base(o) or o.meta['cons']>=4)
    V['both: drop (v<=7, none) and add (all five)']=run(lambda o:(base(o) and not (o.meta['cons']==0 and o.meta['v']<=7)) or o.meta['cons']==5)
    V['add objects in no vote file that >=4 second-opinion files have']=run(base,lambda o:o.meta['cons']>=4)
    st=collections.Counter(('in',o.meta['v'] if o.meta['v']<8 else 8,min(o.meta['cons'],5)) for o in objs if base(o))
    return ph,V,st
if __name__=='__main__':
    with Pool(3) as p: R=p.map(photo,photos,chunksize=2)
    names=list(R[0][1]); rng=np.random.default_rng(5); S=[rng.choice(len(photos),size=90,replace=True) for _ in range(1000)]
    E={n:[r[1][n][0] for r in R] for n in names}
    def pq(n,s=None): return pq_from([c for i in (range(len(photos)) if s is None else s) for c in E[n][i]])['pq']
    bb=np.array([pq(names[0],s) for s in S]); b0=pq(names[0])
    print('%-62s masks   PQ     change   90-photo sd  share positive'%'variant')
    for n in names:
        d=np.array([pq(n,s) for s in S])-bb
        print('%-62s %4d  %.4f  %+.4f   %.4f      %.2f'%(n,sum(r[1][n][1] for r in R),pq(n),pq(n)-b0,d.std(),(d>0).mean()),flush=True)

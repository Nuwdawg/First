"""Label-free: what does the eight-view consensus (max x fraction >= 0.92) drop from, and add to, the single-view
selection (floor 0.85)? Same statistics on validation (by annotator count) and on test."""
import sys,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast2 import paint_labels
from lib import load_train
from cluster import ciou
from multiprocessing import Pool
d,byimg,PH=load_train()
def stats(cands):
    single=paint_labels([c for c in cands if c.meta['view']==0],0.85)[2]
    objs=d4_objects(cands)
    for o in objs: o.score=o.meta['maxfrac']
    cons=paint_labels(objs,0.92)[2]
    def unmatched(A,B):
        out=[]
        for a in A:
            if not any(ciou(a,b)>=0.5 for b in B): out.append(a)
        return out
    def rad(o):
        ys,xs=np.nonzero(o.mask); return float(np.hypot(o.x0+xs.mean()-1024,o.y0+ys.mean()-1024))
    drop=unmatched(single,cons); add=unmatched(cons,single)
    # for dropped objects: the eight-view object they belong to (best IoU among all objects) and its stats
    info=[]
    for a in drop:
        best=max(objs,key=lambda o:ciou(a,o),default=None)
        info.append((a.score,a.area,rad(a),best.meta['n'] if best is not None and ciou(a,best)>=0.5 else 0,best.meta['maxfrac'] if best is not None and ciou(a,best)>=0.5 else 0.0))
    return len(single),len(cons),info,[(o.meta['max'],o.area,rad(o),o.meta['n']) for o in add]
def summ(label,R):
    ns=sum(r[0] for r in R); nc=sum(r[1] for r in R); D=np.array([x for r in R for x in r[2]]).reshape(-1,5); A=np.array([x for r in R for x in r[3]]).reshape(-1,4)
    print('%-22s photos %3d  single %.2f/ph  consensus %.2f/ph | dropped %.2f/ph (%.1f%% of single): score %.3f area %5.0f radius %4.0f views %.2f, share seen in <=4 views %.2f | added %.2f/ph: max score %.3f area %5.0f radius %4.0f views %.2f'%(
        label,len(R),ns/len(R),nc/len(R),len(D)/len(R),100*len(D)/ns,D[:,0].mean(),np.median(D[:,1]),D[:,2].mean(),D[:,3].mean(),(D[:,3]<=4).mean(),len(A)/len(R),A[:,0].mean(),np.median(A[:,1]),A[:,2].mean(),A[:,3].mean()))
if __name__=='__main__':
    V=load('e40','val'); T=load('e40','test')
    with Pool(4) as p:
        RV=dict(zip(sorted(V),p.map(stats,[V[ph] for ph in sorted(V)],chunksize=4)))
        RT=p.map(stats,[T[ph] for ph in sorted(T)],chunksize=4)
    summ('validation, all',list(RV.values()))
    for k in (1,2,3): summ('validation, %d record(s)'%k,[RV[ph] for ph in RV if len(PH[ph])==k])
    summ('test',RT)

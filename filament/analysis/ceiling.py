"""Leave-one-annotator-out ceiling: pick candidates using two annotators' masks, score on the third."""
import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from zoo_eval import *
def cand_iou(c,items):
    out=np.zeros(len(items))
    for i,(x,y,m,ga) in enumerate(items):
        x0=max(x,c.x0); y0=max(y,c.y0); x1=min(x+m.shape[1],c.x0+c.mask.shape[1]); y1=min(y+m.shape[0],c.y0+c.mask.shape[0])
        if x1<=x0 or y1<=y0: continue
        inter=(m[y0-y:y1-y,x0-x:x1-x]&c.mask[y0-c.y0:y1-c.y0,x0-c.x0:x1-c.x0]).sum()
        out[i]=inter/(ga+c.area-inter)
    return out
V=load('e40','val'); photos=sorted(V); G=GT(photos)
multi=[ph for ph in photos if len(G.g[ph])>=2]
print('multi-annotated val photos',len(multi),'records',sum(len(G.g[ph]) for ph in multi))
def m_of(c,items):
    v=cand_iou(c,items); b=v.max() if len(v) else 0
    return b if b>0.5 else 0.0
for lam in (0.15,0.2,0.25,0.3,0.35):
    dep=[];orc=[];orc_all=[]
    for ph in multi:
        cs=[c for c in V[ph] if c.meta['view']==0 and c.area>=100]
        ents=G.g[ph]
        M=np.array([[m_of(c,items) for e,items in ents] for c in cs])   # cand x annotator
        for i,(e,items) in enumerate(ents):
            others=[j for j in range(len(ents)) if j!=i]
            est=M[:,others].mean(1)                                   # estimate from the other annotators
            order=[c for s,c in sorted(zip(est,cs),key=lambda t:-t[0]) if s>lam]
            for c in order: c.score=1.0
            cv,ar,k=paint_labels(order,0.5,order=order)
            orc.append(pq_counts(fast2.iou_mat(cv,ar,items)))
            cv,ar,k=paint_labels(sorted(cs,key=lambda c:-c.meta["p"]*c.meta["m"]),-1,order=[c for c in sorted(cs,key=lambda c:-c.meta["p"]*c.meta["m"]) if c.meta["p"]*c.meta["m"]>=0.85])
            dep.append(pq_counts(fast2.iou_mat(cv,ar,items)))
    a=pq_from(orc); b=pq_from(dep)
    print('lambda %.2f | leave-one-annotator-out oracle PQ %.4f (%d/%d/%d) | deployed on same records %.4f (%d/%d/%d)'%(lam,a['pq'],a['tp'],a['fp'],a['fn'],b['pq'],b['tp'],b['fp'],b['fn']),flush=True)

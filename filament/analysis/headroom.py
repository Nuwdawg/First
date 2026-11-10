import sys,time; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); G=GT(list(C))
def cand_iou(c,items):
    out=np.zeros(len(items))
    for i,(x,y,m,ga) in enumerate(items):
        x0=max(x,c.x0); y0=max(y,c.y0); x1=min(x+m.shape[1],c.x0+c.mask.shape[1]); y1=min(y+m.shape[0],c.y0+c.mask.shape[0])
        if x1<=x0 or y1<=y0: continue
        inter=(m[y0-y:y1-y,x0-x:x1-x]&c.mask[y0-c.y0:y1-c.y0,x0-c.x0:x1-c.x0]).sum()
        out[i]=inter/(ga+c.area-inter)
    return out
rows=[]  # score, area, best pooled iou, n annotators matching, k
for ph,cs in C.items():
    ents=G.g[ph]
    for c in cs:
        if c.area<100: continue
        per=[cand_iou(c,items) for e,items in ents]
        best=[p.max() if len(p) else 0 for p in per]
        rows.append((c.score,c.area,max(best),sum(b>0.5 for b in best),len(ents)))
R=np.array(rows); print('cands',len(R))
print('score bin | n | P(any match) | P(specific match) | E[m_any] | mean IoU|match')
edges=[0,.02,.05,.1,.2,.3,.4,.5,.6,.7,.8,.85,.9,.95,1.01]
for a,b in zip(edges[:-1],edges[1:]):
    s=(R[:,0]>=a)&(R[:,0]<b)
    if s.sum()==0: continue
    anym=R[s,2]>0.5; spec=(R[s,3]/R[s,4]).mean(); em=np.where(anym,R[s,2],0).mean()
    print('%.2f-%.2f %5d  %.3f  %.3f  %.3f  %.3f'%(a,b,s.sum(),anym.mean(),spec,em,R[s,2][anym].mean() if anym.any() else 0))
for k in (1,2,3):
    s=(R[:,4]==k)&(R[:,0]>=0.7)&(R[:,0]<0.85)
    print('k',k,'score .70-.85 n',s.sum(),'P(any)',(R[s,2]>0.5).mean().round(3),'P(spec)',(R[s,3]/R[s,4]).mean().round(3))
# oracle: paint candidates in descending order of true best IoU (only those >0.5), pooled score
ent=[];pool=[];n=0
class W: pass
for ph,cs in C.items():
    ents=G.g[ph]; allitems=[it for e,items in ents for it in items]
    sc=[]
    for c in cs:
        v=cand_iou(c,allitems); sc.append(v.max() if len(v) else 0)
    order=[c for s,c in sorted(zip(sc,cs),key=lambda t:-t[0]) if s>0.5]
    for c in order: c.score=1.0
    cv,ar,k=paint_labels(order,0.5,order=order); e,p=score_photo(cv,ar,ents); ent+=e; pool.append(p); n+=len(ar)
print('ORACLE (pick by true any-annotator IoU, single-view candidates): n',n,'entry',pq_from(ent),'pooled',pq_from(pool))

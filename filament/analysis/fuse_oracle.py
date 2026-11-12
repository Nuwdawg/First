"""Label fusion ceiling on triple-annotated training photographs (labels only, no model).
What single prediction scores best against three annotators' records under the official per-record rule?
Policies: copy one annotator; fuse masks that two or three annotators drew (majority pixels, union, intersection,
medoid); with or without the masks only one annotator drew."""
import sys,numpy as np,collections,itertools; sys.path.insert(0,'/home/user/work/an')
from lib import *
from fast import Cand
from fast2 import paint_labels, iou_mat, H, W
import cluster as CLU
from multiprocessing import Pool
d,byimg,photos=load_train()
P3=sorted(ph for ph,e in photos.items() if len(e)==3)
def items_of(rles):
    out=[]
    for r in rles:
        x,y,w,h=[int(v) for v in mu.toBbox(r)]; m=mu.decode(r)[y:y+h,x:x+w].astype(bool); out.append((x,y,m,int(m.sum())))
    return out
def mk(x0,y0,mask,score,meta):
    o=object.__new__(Cand); o.x0,o.y0,o.mask,o.area,o.score,o.meta=x0,y0,mask,int(mask.sum()),score,meta; return o
def fuse(members,mode):
    x0=min(m.x0 for m in members); y0=min(m.y0 for m in members); x1=max(m.x0+m.mask.shape[1] for m in members); y1=max(m.y0+m.mask.shape[0] for m in members)
    acc=np.zeros((y1-y0,x1-x0),np.int32)
    for m in members: acc[m.y0-y0:m.y0-y0+m.mask.shape[0],m.x0-x0:m.x0-x0+m.mask.shape[1]]+=m.mask
    n=len(members)
    if mode=='union': mask=acc>=1
    elif mode=='inter': mask=acc>=n
    elif mode=='major': mask=acc*2>n if n>2 else acc>=1      # 2 of 3; for two members the union
    elif mode=='major_i': mask=acc*2>n if n>2 else acc>=2    # 2 of 3; for two members the intersection
    elif mode=='medoid':
        if n==2: best=max(members,key=lambda m:m.area)
        else: best=max(members,key=lambda m:sum(CLU.ciou(m,o) for o in members if o is not m))
        return best
    elif mode=='medoid_small':
        if n==2: best=min(members,key=lambda m:m.area)
        else: best=max(members,key=lambda m:sum(CLU.ciou(m,o) for o in members if o is not m))
        return best
    return mk(x0,y0,mask,1.0,{})
def photo(ph):
    ents=photos[ph]; R={e:items_of(entry_rles(byimg,e)) for e in ents}
    pool=[mk(x,y,m,float(a),{'src':e}) for e in ents for (x,y,m,a) in R[e]]
    clusters=CLU.cluster(pool,link=0.5,min_score=0.0)
    out={}
    def score(objs):
        objs=[o for o in objs if o.area>=1]
        cv,ar,k=paint_labels(objs,-1,min_area=1,order=objs)
        return [pq_counts(iou_mat(cv,ar,R[e])) for e in ents]
    out['copy annotator 1']=score([mk(x,y,m,1.0,{}) for (x,y,m,a) in R[ents[0]]])
    out['copy annotator 2']=score([mk(x,y,m,1.0,{}) for (x,y,m,a) in R[ents[1]]])
    out['copy annotator 3']=score([mk(x,y,m,1.0,{}) for (x,y,m,a) in R[ents[2]]])
    multi=sorted([c for c in clusters if len(c.members)>=2],key=lambda c:(-len(c.members),-c.seed.area)); single=[c for c in clusters if len(c.members)==1]
    for mode in ('major','major_i','union','inter','medoid','medoid_small'):
        fused=[fuse(c.members,mode) for c in multi]
        out['fused>=2 %s'%mode]=score(fused)
        out['fused>=2 %s + singles'%mode]=score(fused+[c.seed for c in single])
    three=[c for c in multi if len(c.members)==3]
    out['fused=3 major']=score([fuse(c.members,'major') for c in three])
    st=collections.Counter(len(c.members) for c in clusters)
    return out,st,sum(len(R[e]) for e in ents)
if __name__=='__main__':
    with Pool(4) as p: RES=p.map(photo,P3,chunksize=4)
    tot=collections.Counter(); ngt=0
    for o,st,n in RES: tot.update(st); ngt+=n
    print('triple-annotated photographs:',len(P3),' annotated masks:',ngt,' clusters by number of annotators:',dict(sorted(tot.items())))
    print('  share of masks in clusters of 1/2/3 annotators: %.3f %.3f %.3f'%(tot[1]/ngt,2*tot[2]/ngt,3*tot[3]/ngt))
    for k in RES[0][0]:
        a=pq_from([c for o,st,n in RES for c in o[k]])
        print('%-36s PQ %.4f  sq %.3f rq %.3f  (%d/%d/%d)'%(k,a['pq'],a['sq'],a['rq'],a['tp'],a['fp'],a['fn']))

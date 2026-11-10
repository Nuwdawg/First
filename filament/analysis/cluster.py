"""Multi-view / multi-source clustering of candidates into consensus objects."""
import sys; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
def ciou(a,b):
    x0=max(a.x0,b.x0); y0=max(a.y0,b.y0); x1=min(a.x0+a.mask.shape[1],b.x0+b.mask.shape[1]); y1=min(a.y0+a.mask.shape[0],b.y0+b.mask.shape[0])
    if x1<=x0 or y1<=y0: return 0.0
    inter=int((a.mask[y0-a.y0:y1-a.y0,x0-a.x0:x1-a.x0]&b.mask[y0-b.y0:y1-b.y0,x0-b.x0:x1-b.x0]).sum())
    return inter/(a.area+b.area-inter) if inter else 0.0
class Cluster:
    __slots__=('members','seed')
    def __init__(self,c): self.members=[c]; self.seed=c
def cluster(cands,link=0.5,min_score=0.02):
    cl=[]
    for c in sorted(cands,key=lambda c:-c.score):
        if c.score<min_score or c.area<20: continue
        best=None;bi=link
        for k in cl:
            # one member per source-view per cluster
            if any(m.meta['src']==c.meta['src'] for m in k.members): continue
            v=ciou(k.seed,c)
            if v>=bi: bi=v;best=k
        if best is None: cl.append(Cluster(c))
        else: best.members.append(c)
    return cl
def consensus(k,nsrc,mode='vote',thr=0.5):
    """returns a Cand-like object with consensus mask and mean score over nsrc sources"""
    ms=k.members
    x0=min(m.x0 for m in ms); y0=min(m.y0 for m in ms); x1=max(m.x0+m.mask.shape[1] for m in ms); y1=max(m.y0+m.mask.shape[0] for m in ms)
    acc=np.zeros((y1-y0,x1-x0),np.float32); wsum=0
    for m in ms:
        w=m.score if mode=='wvote' else 1.0
        acc[m.y0-y0:m.y0-y0+m.mask.shape[0],m.x0-x0:m.x0-x0+m.mask.shape[1]]+=w*m.mask; wsum+=w
    mask=acc>=thr*wsum if mode!='seed' else None
    o=object.__new__(Cand)
    if mode=='seed':
        s=k.seed; o.x0,o.y0,o.mask=s.x0,s.y0,s.mask
    else:
        o.x0,o.y0,o.mask=x0,y0,mask
    o.area=int(o.mask.sum()); o.meta={'n':len(ms),'max':max(m.score for m in ms),'sum':sum(m.score for m in ms)}
    o.score=o.meta['sum']/nsrc
    return o

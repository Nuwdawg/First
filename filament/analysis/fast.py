"""Fast candidate handling: cropped boolean masks, painting, and both scoring protocols."""
import sys; sys.path.insert(0,'/home/user/work/an')
from lib import *
import json, pickle, os
def rle_fix(r):
    r=dict(r)
    if isinstance(r['counts'],str): r['counts']=r['counts'].encode()
    return r
class Cand:
    __slots__=('score','p','m','x0','y0','mask','area','meta')
    def __init__(self,rle,score,meta=None):
        r=rle_fix(rle); bb=mu.toBbox(r); x,y,w,h=[int(v) for v in bb]
        full=mu.decode(r)
        self.x0,self.y0=x,y; self.mask=full[y:y+h,x:x+w].astype(bool); self.area=int(self.mask.sum()); self.score=score; self.meta=meta
def to_rle(mask,x0,y0):
    full=np.zeros((H,W),np.uint8); h,w=mask.shape; full[y0:y0+h,x0:x0+w]=mask
    return mu.encode(np.asfortranarray(full))
def paint(cands,floor,min_area=100,key=None):
    """descending-score painting into a label canvas; returns list of RLEs and kept cands"""
    canvas=np.zeros((H,W),bool); out=[]; kept=[]
    for c in sorted(cands,key=key or (lambda c:-c.score)):
        if c.score<floor: break
        h,w=c.mask.shape; sub=canvas[c.y0:c.y0+h,c.x0:c.x0+w]
        m=c.mask&~sub
        a=int(m.sum())
        if a<min_area: continue
        sub|=m
        out.append(to_rle(m,c.x0,c.y0)); kept.append(c)
    return out,kept
class Scorer:
    def __init__(self,photo_list):
        d,byimg,photos=load_train()
        self.gt={ph:[entry_rles(byimg,e) for e in photos[ph]] for ph in photo_list}
        self.ents={ph:photos[ph] for ph in photo_list}
        self.pool={ph:[x for g in self.gt[ph] for x in g] for ph in photo_list}
    def score(self,preds,per_photo=False):
        ent=[];pool=[];pp={}
        for ph,pr in preds.items():
            e=[pq_counts(iou_matrix(g,pr)) for g in self.gt[ph]]
            p=greedy_one_to_one(iou_matrix(self.pool[ph],pr))
            ent+=e; pool.append(p); pp[ph]=(e,p)
        return (pq_from(ent),pq_from(pool),pp) if per_photo else (pq_from(ent),pq_from(pool))
def load_val_cands(kind='full'):
    pk=f'/home/user/work/an/val_cands_{kind}.pkl'
    if os.path.exists(pk): return pickle.load(open(pk,'rb'))
    vc=json.load(open('/home/user/work/caches/exp147/val_cache.json'))['photographs']
    out={}
    for fn,v in vc.items():
        out[fn.split('.')[0]]=[Cand(c['rle'],c['score'],{k:c[k] for k in c if k not in('rle','rle_low','rle_high','embedding')}) for c in v[kind]]
    pickle.dump(out,open(pk,'wb'))
    return out

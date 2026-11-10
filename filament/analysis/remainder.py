"""Painting policy: what to do with a candidate partly covered by higher-scoring ones."""
import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects
import fast2
def paint_policy(cands,floor,min_area=100,min_keep=0.0):
    canvas=np.zeros((H,W),np.int32); areas=[]; kept=[]
    for c in sorted(cands,key=lambda c:-c.score):
        if c.score<floor: break
        h,w=c.mask.shape; sub=canvas[c.y0:c.y0+h,c.x0:c.x0+w]
        m=c.mask&(sub==0); a=int(m.sum())
        if a<min_area or a<min_keep*c.area: continue
        sub[m]=len(areas)+1; areas.append(a); kept.append(c)
    return canvas,np.array(areas,np.int64),kept
V=load('e40','val'); photos=sorted(V); G=GT(photos)
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
S={ph:[c for c in V[ph] if c.meta['view']==0] for ph in photos}
O={ph:d4_objects(V[ph]) for ph in photos}
for ph in photos:
    for o in O[ph]: o.score=o.meta['mean8']
def runp(objs,floor,mk):
    res={}; n=0; part=0
    for ph in photos:
        cv,ar,k=paint_policy(objs[ph],floor,min_keep=mk); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[ph]=e; n+=len(ar); part+=sum(1 for a,c in zip(ar,k) if a<c.area)
    return res,n,part
for label,objs,floor in [('single @0.85',S,0.85),('d4 mean8 @0.85',O,0.85)]:
    for mk in (0.0,0.3,0.5,0.7,0.9):
        r,n,part=runp(objs,floor,mk); a=pq_of(r); d_,ci,pr=boot(r,base,photos,n=400)
        print('%-16s keep-fraction>=%.1f: PQ %.4f (%d/%d/%d) n %d, partly-covered kept %d | vs deployed %+.4f'%(label,mk,a['pq'],a['tp'],a['fp'],a['fn'],n,part,d_),flush=True)

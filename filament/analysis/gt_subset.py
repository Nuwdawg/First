"""Would a filtered test ground truth (by chirality class or by area) reproduce the leaderboard pattern 0.39 / 0.41 / 0.40 at floors 0.70 / 0.85 / 0.90?"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); photos=list(C)
d,byimg,PH=load_train()
def gt_items(keep):
    g={}
    for ph in photos:
        L=[]
        for e in PH[ph]:
            items=[]
            for a in byimg[e]:
                if not keep(a): continue
                r=ann_rle(a); x,y,w,h=[int(v) for v in mu.toBbox(r)]; m=mu.decode(r)[y:y+h,x:x+w].astype(bool); items.append((x,y,m,int(m.sum())))
            L.append((e,items))
        g[ph]=L
    return g
def pq_at(g,floor,min_area=100):
    ent=[]
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,floor,min_area=min_area)
        for e,items in g[ph]:
            if len(items)==0 and True: ent.append((0.0,0,len(ar),0)); continue
            ent.append(pq_counts(iou_mat(cv,ar,items)))
    return pq_from(ent)
cats=collections.Counter(a['category_id'] for ph in photos for e in PH[ph] for a in byimg[e]); print('validation annotations by class:',dict(cats))
print('%-34s %7s | %8s %8s %8s %8s | best floor'%('ground truth kept','n GT','0.70','0.85','0.90','0.95'))
for label,keep in [('all (as validated)',lambda a:True),('Left/Right only (classes 1,2)',lambda a:a['category_id'] in (1,2)),('Unidentifiable only (class 3)',lambda a:a['category_id']==3),('area >= 300',lambda a:a['area']>=300),('area >= 600',lambda a:a['area']>=600),('area >= 1000',lambda a:a['area']>=1000),('area < 3000',lambda a:a['area']<3000)]:
    g=gt_items(keep); n=sum(len(items) for ph in photos for e,items in g[ph])
    r={f:pq_at(g,f)['pq'] for f in (0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97)}
    print('%-34s %7d | %8.4f %8.4f %8.4f %8.4f | %.2f (%.4f)'%(label,n,r[0.7],r[0.85],r[0.9],r[0.95],max(r,key=r.get),max(r.values())),flush=True)

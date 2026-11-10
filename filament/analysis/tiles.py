"""Do native 1024-px tile views (stride 512) add to e40's eight-view consensus on validation?"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects
V=load('e40','val'); photos=sorted(V); G=GT(photos); _,_,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
TI=load_val_cands('tiles'); base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
POS=[0,512,1024]
def covering(o):
    """tiles that contain the object's bounding box with an 8-px margin"""
    x0,y0=o.x0,o.y0; x1=x0+o.mask.shape[1]; y1=y0+o.mask.shape[0]
    return [(tx,ty) for tx in POS for ty in POS if x0>=tx+8*(tx>0) and y0>=ty+8*(ty>0) and x1<=tx+1024-8*(tx<1024) and y1<=ty+1024-8*(ty<1024)]
def report(label,O,FL):
    rows=[]
    for f in FL:
        r=run(O,G,photos,f); rows.append((pq_of(r)['pq'],f,r,pq_from([c for p in photos if kk[p]>=2 for c in r[p]])['pq']))
    b=max(rows,key=lambda t:t[0]); d_,ci,pr=boot(b[2],base,photos,n=500)
    print('%-40s best %.4f @%.2f %+.4f [%+.4f,%+.4f] | %s | k>=2 best %.4f'%(label,b[0],b[1],d_,ci[0],ci[1],' '.join('%.2f:%.4f'%(f,p) for p,f,_,_ in rows),max(r[3] for r in rows)),flush=True)
O={}; stats=[]
for ph in photos:
    objs=d4_objects(V[ph]); tl=[t for t in TI[ph] if not t.meta.get('edges')]
    for o in objs:
        cov=covering(o); ts=[]
        for (tx,ty) in cov:
            best=0.0
            for t in tl:
                if t.meta['x0']==tx and t.meta['y0']==ty and ciou(o,t)>=0.5: best=max(best,t.score)
            ts.append(best)
        o.meta['tile_n']=len(cov); o.meta['tile_mean']=float(np.mean(ts)) if ts else None; o.meta['tile_frac']=float(np.mean([s>0 for s in ts])) if ts else None
    O[ph]=objs
cov=[o.meta['tile_n'] for ph in photos for o in O[ph] if o.meta['mean8']>=0.5]; print('objects with mean8>=0.5: %d; covered by >=1 tile: %.2f; mean tiles covering %.2f'%(len(cov),np.mean([c>0 for c in cov]),np.mean(cov)))
def setscore(fn):
    for ph in photos:
        for o in O[ph]: o.score=fn(o.meta)
setscore(lambda m:m['mean8']); report('d4 mean8 (reference)',O,(0.8,0.85,0.88,0.9))
setscore(lambda m:m['maxfrac']); report('d4 maxfrac (reference)',O,(0.88,0.9,0.92,0.94))
setscore(lambda m:m['tile_mean'] if m['tile_mean'] is not None else m['mean8']); report('tile mean alone (fallback mean8)',O,(0.6,0.7,0.75,0.8,0.85,0.88,0.9))
for w in (0.25,0.5):
    setscore(lambda m,w=w:(1-w)*m['mean8']+w*(m['tile_mean'] if m['tile_mean'] is not None else m['mean8'])); report('mean8 blended with tile mean, w=%.2f'%w,O,(0.75,0.8,0.85,0.88,0.9))
    setscore(lambda m,w=w:(1-w)*m['maxfrac']+w*(m['tile_mean'] if m['tile_mean'] is not None else m['maxfrac'])); report('maxfrac blended with tile mean, w=%.2f'%w,O,(0.8,0.85,0.88,0.9,0.92))
setscore(lambda m:m['maxfrac']*(m['tile_frac'] if m['tile_frac'] is not None else 1.0)); report('maxfrac x tile fraction',O,(0.85,0.88,0.9,0.92,0.94))

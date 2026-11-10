import sys,time; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); G=GT(list(C))
d,byimg,photos=load_train()
# val record density
dens={e:len(items) for ph in C for e,items in G.g[ph]}
vals=np.array(list(dens.values())); print('val records',len(vals),'GT/record mean %.2f median %.1f'%(vals.mean(),np.median(vals)))
print('model detections per photo @0.85: val',np.mean([sum(c.score>=0.85 for c in C[ph]) for ph in C]).round(2))
def run(floor,min_area):
    out={}
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,floor,min_area=min_area); e,p=score_photo(cv,ar,G.g[ph])
        for (ename,items),cc in zip(G.g[ph],e): out[ename]=cc
    return out
def agg(out,sel): return pq_from([out[e] for e in out if sel(e)])
print('floor area | all | sparse recs (<=4 GT) | mid (5-8) | dense (>=9)   [n recs: %d %d %d]'%(sum(v<=4 for v in vals),sum((v>=5)&(v<=8) for v in vals),sum(v>=9 for v in vals)))
for floor in [0.8,0.85,0.9,0.93,0.95]:
    for ma in [100,200,300,500,800]:
        o=run(floor,ma)
        a=agg(o,lambda e:True); s=agg(o,lambda e:dens[e]<=4); m=agg(o,lambda e:5<=dens[e]<=8); dd=agg(o,lambda e:dens[e]>=9)
        print('%.2f %4d | %.4f (%d/%d/%d) | %.4f (%d/%d/%d) | %.4f | %.4f'%(floor,ma,a['pq'],a['tp'],a['fp'],a['fn'],s['pq'],s['tp'],s['fp'],s['fn'],m['pq'],dd['pq']),flush=True)

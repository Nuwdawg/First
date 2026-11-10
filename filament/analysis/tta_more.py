import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
V=load('e40','val'); photos=sorted(V); G=GT(photos)
t0=time.time()
SC=load_val_cands('scale'); TI=load_val_cands('tiles'); print('loaded scale',sum(len(v) for v in SC.values()),'tiles',sum(len(v) for v in TI.values()),round(time.time()-t0),flush=True)
for ph in photos:
    for c in SC[ph]: c.meta['src']='e40:scale'
    for c in TI[ph]: c.meta['src']='e40:tile%d_%d'%(c.meta.get('x0',0),c.meta.get('y0',0))
# scale alone
S={ph:SC[ph] for ph in photos}
print('scale-2560 single view:',' '.join('%.2f:%.4f'%(f,pq_of(run(S,G,photos,f))['pq']) for f in (0.75,0.8,0.85,0.88,0.9)),flush=True)
# d4 + scale (9 sources)
CL={ph:cluster(V[ph]+SC[ph],link=0.5) for ph in photos}
O={ph:[consensus(k,9,'vote',0.6) for k in cl] for ph,cl in CL.items()}
evalset(O,G,photos,9,'e40 d4+scale')
# full + scale only (2 sources)
CL={ph:cluster([c for c in V[ph] if c.meta['view']==0]+SC[ph],link=0.5) for ph in photos}
O={ph:[consensus(k,2,'vote',0.5) for k in cl] for ph,cl in CL.items()}
evalset(O,G,photos,2,'e40 full+scale',FL=(0.6,0.7,0.75,0.8,0.85,0.88,0.9))

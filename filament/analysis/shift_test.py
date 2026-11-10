import sys,time,pickle,json; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
d=json.load(open('/home/user/work/caches/codex-swins-heldout-views/swins/swins_val_cache.json'))['photographs']
V={}
for fn,v in d.items():
    L=[]
    for mode in ('full','d4','shift'):
        for c in v.get(mode,[]):
            src='full' if mode=='full' else ('d4_%s'%c['view'] if mode=='d4' else 'sh_%s'%c['view'])
            L.append(Cand(c['rle'],c['score'],{'src':src,'mode':mode}))
    V[fn.split('.')[0]]=L
photos=sorted(V); G=GT(photos)
_,_,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
def sweep(O,FL=(0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.94)):
    rows=[]
    for f in FL:
        r=run(O,G,photos,f); rows.append((pq_of(r)['pq'],pq_from([c for ph in photos if kk[ph]>=2 for c in r[ph]])['pq'],f))
    return rows
def show(label,rows): b=max(rows); print('%-26s best all %.4f @%.2f | best k>=2 %.4f | %s'%(label,b[0],b[2],max(r[1] for r in rows),' '.join('%.2f:%.4f'%(r[2],r[0]) for r in rows)),flush=True)
show('swins single',sweep({ph:[c for c in V[ph] if c.meta['mode']=='full'] for ph in photos}))
for label,modes,ns in [('d4 (8 views)',('full','d4'),8),('shift only (4 views)',('full','shift'),4),('d4+shift (11 views)',('full','d4','shift'),11)]:
    CL={ph:cluster([c for c in V[ph] if c.meta['mode'] in modes],link=0.5) for ph in photos}
    O={ph:[consensus(k,ns,'vote',0.6) for k in cl] for ph,cl in CL.items()}
    for agg in ('mean','maxfrac'):
        for ph in photos:
            for o in O[ph]: o.score=o.meta['sum']/ns if agg=='mean' else o.meta['max']*o.meta['n']/ns
        show('swins '+label+' '+agg,sweep(O))

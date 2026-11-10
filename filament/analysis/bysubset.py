import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
V=load('e40','val'); photos=sorted(V); G=GT(photos)
d,byimg,PH=load_train()
k={ph:len(PH[ph]) for ph in photos}
def split(res):
    out={}
    for name,sel in [('k=1',lambda ph:k[ph]==1),('k>=2',lambda ph:k[ph]>=2),('k=3',lambda ph:k[ph]==3),('group01',None)]:
        if name=='group01':
            cs=[c for ph in photos for (e,items),c in zip(G.g[ph],res[ph]) if e.startswith('01')]
        else:
            cs=[c for ph in photos if sel(ph) for c in res[ph]]
        out[name]=pq_from(cs)
    return out
S={ph:[c for c in V[ph] if c.meta['view']==0] for ph in photos}
CL={ph:cluster(V[ph],link=0.5) for ph in photos}
O={ph:[consensus(kk,8,'vote',0.6) for kk in cl] for ph,cl in CL.items()}
def show(label,res):
    s=split(res); a=pq_of(res)
    print('%-28s all %.4f | '%(label,a['pq'])+' | '.join('%s %.4f (n tp %d)'%(n,v['pq'],v['tp']) for n,v in s.items()),flush=True)
for f in (0.8,0.85,0.9): show('single @%.2f'%f,run(S,G,photos,f))
for ph in photos:
    for o in O[ph]: o.score=o.meta['sum']/8
for f in (0.8,0.85,0.88,0.9): show('d4 mean8 @%.2f'%f,run(O,G,photos,f))
for ph in photos:
    for o in O[ph]: o.score=o.meta['max']*o.meta['n']/8
for f in (0.88,0.9,0.92,0.94): show('d4 max*frac @%.2f'%f,run(O,G,photos,f))
print('records: k=1',sum(k[p]==1 for p in photos),'photos; k>=2',sum(k[p]>=2 for p in photos),'photos')

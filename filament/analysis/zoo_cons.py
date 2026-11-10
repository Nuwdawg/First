import sys,time,pickle,itertools; sys.path.insert(0,'/home/user/work/an')
from zoo_eval import *
names=sys.argv[1].split(',')
views=sys.argv[2] if len(sys.argv)>2 else 'full'      # full | d4
V={n:load(n,'val') for n in names}
photos=sorted(V[names[0]]); G=GT(photos)
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
for n in ('base20','qual31'):
    if n in V:
        s=np.array([c.score for ph in photos for c in V[n][ph] if c.meta['view']==0]); print(n,'score quantiles',np.quantile(s,[.5,.75,.9,.95,.99,1]).round(3))
def sources(ph):
    L=[]
    for n in names:
        for c in V[n][ph]:
            if views=='full' and c.meta['view']!=0: continue
            L.append(c)
    return L
nsrc=len({c.meta['src'] for ph in photos[:20] for c in sources(ph)})
print('models',names,'views',views,'sources',nsrc,flush=True)
t0=time.time()
CL={ph:cluster(sources(ph),link=0.5) for ph in photos}; print('clustered',round(time.time()-t0),flush=True)
FL=[0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.88,0.9]
for mode,thr in [('vote',0.5),('vote',0.6)]:
    O={ph:[consensus(k,nsrc,mode,thr) for k in cl] for ph,cl in CL.items()}
    for agg in ['mean','max*frac']:
        for ph in photos:
            for o in O[ph]: o.score=o.meta['sum']/nsrc if agg=='mean' else o.meta['max']*o.meta['n']/nsrc
        rows=[]
        for f in FL:
            r=run(O,G,photos,f); a=pq_of(r); rows.append((a['pq'],f,r,a))
        best=max(rows,key=lambda t:t[0]); d,ci,pr=boot(best[2],base,photos,n=600)
        print('%s%.1f %-8s best %.4f @%.2f (%d/%d/%d) vs deployed %+.4f [%+.4f,%+.4f] P>0 %.2f | %s'%(mode,thr,agg,best[0],best[1],best[3]['tp'],best[3]['fp'],best[3]['fn'],d,ci[0],ci[1],pr,' '.join('%.2f:%.4f'%(f,p) for p,f,_,_ in rows)),flush=True)
print('time',round(time.time()-t0))

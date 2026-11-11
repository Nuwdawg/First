"""Add the two models with a different score scale (base20, qual31) as fifth and sixth members, after rescaling their scores."""
import sys,pickle,time; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
import cluster as CLU
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb')),'ctrl31':load('ctrl31','val'),'base20':load('base20','val'),'qual31':load('qual31','val')}
SCALE={'base20':0.85/0.461,'qual31':0.85/0.469}
photos=sorted(M['e40']); G=GT(photos); base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
S1={}
for name in M:
    S1[name]={}
    for ph in photos:
        objs=[CLU.consensus(k,8,'vote',0.6) for k in CLU.cluster(M[name][ph],link=0.5)]
        f=SCALE.get(name,1.0)
        for o in objs: o.meta['mean8']=min(o.meta['sum']/8*f,1.0); o.meta['maxfrac']=min(o.meta['max']*f,1.0)*o.meta['n']/8; o.meta['src']=name
        S1[name][ph]=objs
fam={}
for subset in [('e40','swins','swinb'),('e40','swins','swinb','ctrl31'),('e40','swins','swinb','ctrl31','qual31'),('e40','swins','swinb','ctrl31','base20'),('e40','swins','swinb','ctrl31','base20','qual31'),('e40','swins','swinb','qual31')]:
    for agg in ('mean8','maxfrac'):
        O={}
        for ph in photos:
            pool=[]
            for n in subset:
                for o in S1[n][ph]: o.score=o.meta[agg]; pool.append(o)
            objs=[CLU.consensus(k,len(subset),'vote',0.5) for k in CLU.cluster(pool,link=0.5,min_score=0.01)]
            for o in objs: o.score=o.meta['sum']/len(subset)
            O[ph]=objs
        for f in ((0.8,0.85,0.88,0.9) if agg=='mean8' else (0.86,0.88,0.9,0.92,0.94)):
            fam[(subset,agg,f)]=run(O,G,photos,f)
T={c:np.array([np.array(r[ph],float).sum(0) for ph in photos]) for c,r in fam.items()}; TB=np.array([np.array(base[ph],float).sum(0) for ph in photos])
pq=lambda Mx:Mx[:,0].sum()/(Mx[:,1].sum()+0.5*Mx[:,2].sum()+0.5*Mx[:,3].sum())
cfgs=list(T); A=np.stack([T[c] for c in cfgs])
for subset in sorted({c[0] for c in cfgs},key=len):
    idx=[i for i,c in enumerate(cfgs) if c[0]==subset]; gs=[]
    for rep in range(300):
        perm=np.random.default_rng(rep).permutation(len(photos)); folds=np.array_split(perm,5); num=np.zeros(4); numb=np.zeros(4)
        for k in range(5):
            tr=np.concatenate([folds[j] for j in range(5) if j!=k]); te=folds[k]
            s=A[idx][:,tr,:].sum(1); sc=s[:,0]/(s[:,1]+0.5*s[:,2]+0.5*s[:,3]); b=idx[int(sc.argmax())]
            num+=A[b,te,:].sum(0); numb+=TB[te].sum(0)
        g=lambda v:v[0]/(v[1]+0.5*v[2]+0.5*v[3]); gs.append(g(num)-g(numb))
    best=max(idx,key=lambda i:pq(T[cfgs[i]]))
    print('%-46s nested gain %+.4f (sd %.4f) | best in-sample %.4f %s'%('+'.join(subset),np.mean(gs),np.std(gs),pq(T[cfgs[best]]),cfgs[best][1:]),flush=True)

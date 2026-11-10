"""Nested cross-validation over a broader consensus family: model subsets, link threshold, stage-1 vote, aggregator, floor."""
import sys,pickle,time,itertools,json; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
import cluster as CLU
Z='/home/user/work/an/zoo'; t0=time.time()
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb')),'ctrl31':load('ctrl31','val')}
photos=sorted(M['e40']); G=GT(photos); base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def stage1(name,link,vote):
    out={}
    for ph in photos:
        objs=[CLU.consensus(k,8,'vote',vote) for k in CLU.cluster(M[name][ph],link=link)]
        for o in objs: o.meta['mean8']=o.meta['sum']/8; o.meta['maxfrac']=o.meta['max']*o.meta['n']/8; o.meta['src']=name
        out[ph]=objs
    return out
fam={}; S1={}
for link in (0.4,0.5,0.6):
    for vote in (0.5,0.6):
        for name in M: S1[(name,link,vote)]=stage1(name,link,vote)
        print('stage1',link,vote,round(time.time()-t0),flush=True)
        for subset in [('e40',),('e40','swins'),('e40','swinb'),('e40','swins','swinb'),('e40','swins','swinb','ctrl31'),('swins','swinb')]:
            for agg in ('mean8','maxfrac'):
                O={}
                for ph in photos:
                    pool=[]
                    for n in subset:
                        for o in S1[(n,link,vote)][ph]: o.score=o.meta[agg]; pool.append(o)
                    if len(subset)==1: objs=pool
                    else:
                        objs=[CLU.consensus(k,len(subset),'vote',0.5) for k in CLU.cluster(pool,link=link,min_score=0.01)]
                        for o in objs: o.score=o.meta['sum']/len(subset)
                    O[ph]=objs
                for f in ((0.8,0.85,0.88,0.9) if agg=='mean8' else (0.88,0.9,0.92,0.94)):
                    fam[(subset,link,vote,agg,f)]=run(O,G,photos,f)
        pickle.dump(fam,open('/home/user/work/an/family_cv.pkl','wb'))
T={c:np.array([np.array(r[ph],float).sum(0) for ph in photos]) for c,r in fam.items()}; TB=np.array([np.array(base[ph],float).sum(0) for ph in photos])
pq=lambda Mx:Mx[:,0].sum()/(Mx[:,1].sum()+0.5*Mx[:,2].sum()+0.5*Mx[:,3].sum())
ins=sorted(((pq(T[c]),c) for c in T),reverse=True)
lines=['configs: %d'%len(T),'top 12 in-sample:']+['  %.4f %s'%(p,c) for p,c in ins[:12]]
cfgs=list(T); A=np.stack([T[c] for c in cfgs])   # cfg x photo x 4
gains=[];picks=collections.Counter()
for rep in range(300):
    perm=np.random.default_rng(rep).permutation(len(photos)); folds=np.array_split(perm,5); num=np.zeros(4); numb=np.zeros(4)
    for k in range(5):
        tr=np.concatenate([folds[j] for j in range(5) if j!=k]); te=folds[k]
        s=A[:,tr,:].sum(1); sc=s[:,0]/(s[:,1]+0.5*s[:,2]+0.5*s[:,3]); b=int(sc.argmax()); picks[cfgs[b]]+=1
        num+=A[b,te,:].sum(0); numb+=TB[te].sum(0)
    g=lambda v:v[0]/(v[1]+0.5*v[2]+0.5*v[3]); gains.append(g(num)-g(numb))
gains=np.array(gains); lines+=['nested 5-fold CV over the whole family (300 repeats): mean gain vs deployed %+.4f (sd %.4f, min %+.4f, max %+.4f)'%(gains.mean(),gains.std(),gains.min(),gains.max()),'most often picked:']+['  %d %s'%(n,c) for c,n in picks.most_common(8)]
# restricted families
for label,keep in [('three-model only',lambda c:c[0]==('e40','swins','swinb')),('three-model, link .5, vote .6',lambda c:c[0]==('e40','swins','swinb') and c[1]==0.5 and c[2]==0.6),('e40 only',lambda c:c[0]==('e40',)),('four-model only',lambda c:len(c[0])==4)]:
    idx=[i for i,c in enumerate(cfgs) if keep(c)]; gs=[]
    for rep in range(300):
        perm=np.random.default_rng(rep).permutation(len(photos)); folds=np.array_split(perm,5); num=np.zeros(4); numb=np.zeros(4)
        for k in range(5):
            tr=np.concatenate([folds[j] for j in range(5) if j!=k]); te=folds[k]
            s=A[idx][:,tr,:].sum(1); sc=s[:,0]/(s[:,1]+0.5*s[:,2]+0.5*s[:,3]); b=idx[int(sc.argmax())]
            num+=A[b,te,:].sum(0); numb+=TB[te].sum(0)
        g=lambda v:v[0]/(v[1]+0.5*v[2]+0.5*v[3]); gs.append(g(num)-g(numb))
    lines.append('nested CV, %-32s: %+.4f (sd %.4f); best in-sample %.4f'%(label,np.mean(gs),np.std(gs),max(pq(T[cfgs[i]]) for i in idx)))
open('/home/user/work/an/family_cv.txt','w').write('\n'.join(lines)+'\n'); print('\n'.join(lines)); print('seconds',round(time.time()-t0))

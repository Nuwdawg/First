"""Upper bound on knowing the annotator: oracle confidence floor per annotator batch / per record; and nested CV of the consensus rule family."""
import sys,pickle,itertools; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos); _,_,PH=load_train()
FL=[0.5,0.6,0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.94,0.96]
def per_record(objs_by_photo,floor):
    out={}
    for ph in photos:
        cv,ar,k=paint_labels(objs_by_photo[ph],floor); e,p=score_photo(cv,ar,G.g[ph])
        for (en,items),c in zip(G.g[ph],e): out[en]=c
    return out
def oracle(label,objs):
    R={f:per_record(objs,f) for f in FL}
    recs=list(R[FL[0]])
    glob=max(FL,key=lambda f:pq_from(list(R[f].values()))['pq'])
    print('%-22s global best floor %.2f PQ %.4f'%(label,glob,pq_from(list(R[glob].values()))['pq']))
    for name,key in [('per annotator batch (6-digit prefix)',lambda e:e.split('-')[0]),('per annotator group (4 digits)',lambda e:e[:4]),('per photo',lambda e:e.split('-',1)[1]),('per record',lambda e:e)]:
        groups={}
        for e in recs: groups.setdefault(key(e),[]).append(e)
        chosen=[]
        for g,es in groups.items():
            bf=max(FL,key=lambda f:pq_from([R[f][e] for e in es])['pq']); chosen+=[R[bf][e] for e in es]
        a=pq_from(chosen); print('   oracle floor %-38s (%3d groups): PQ %.4f (%d/%d/%d)'%(name,len(groups),a['pq'],a['tp'],a['fp'],a['fn']),flush=True)
    return R
S={ph:[c for c in M['e40'][ph] if c.meta['view']==0] for ph in photos}
oracle('e40 single view',S)
O3={ph:multi_objects({m:M[m][ph] for m in M},'mean8') for ph in photos}
R3=oracle('three-model mean8',O3)
# nested CV over the rule family for the three-model consensus: choose (agg, floor) on 4/5 of photos, score on 1/5
fam={}
for agg in ('mean8','maxfrac'):
    Oa={ph:multi_objects({m:M[m][ph] for m in M},agg) for ph in photos}
    for f in (0.8,0.85,0.88,0.9,0.92,0.94):
        res={}
        for ph in photos:
            cv,ar,k=paint_labels(Oa[ph],f); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e
        fam[(agg,f)]=res
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def pqs(res,idx): return pq_from([c for i in idx for c in res[photos[i]]])['pq']
gains=[];picks=collections.Counter()
for rep in range(200):
    perm=np.random.default_rng(rep).permutation(len(photos)); folds=np.array_split(perm,5); num=np.zeros(4); numb=np.zeros(4)
    for k in range(5):
        tr=np.concatenate([folds[j] for j in range(5) if j!=k]); te=folds[k]
        best=max(fam,key=lambda c:pqs(fam[c],tr)); picks[best]+=1
        for i in te:
            for c in fam[best][photos[i]]: num+=np.array(c)
            for c in base[photos[i]]: numb+=np.array(c)
    pq=lambda v:v[0]/(v[1]+0.5*v[2]+0.5*v[3]); gains.append(pq(num)-pq(numb))
gains=np.array(gains); print('nested 5-fold CV (200 repeats) of three-model rule family vs deployed: mean gain %+.4f, sd over repeats %.4f, min %+.4f, max %+.4f'%(gains.mean(),gains.std(),gains.min(),gains.max()))
print('configs picked:',picks.most_common(6))
print('in-sample scores:',{k:round(pq_of(v)['pq'],4) for k,v in fam.items()})

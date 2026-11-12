"""Which existing model suits the annotators who score the test set? Single-view and eight-view PQ per model,
by annotator population (target = groups 0101/0102/0104/0501; k=3 = triple-annotated photographs)."""
import sys,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load, ZOO
from fast2 import GT, score_photo, paint_labels
from lib import load_train, pq_from
from multiprocessing import Pool
Z='/home/user/work/an/zoo'; TARGET={'0101','0102','0104','0501'}
NAMES=['e40','ctrl31','ctrl7','focal','head','paste','swins','swinb']
def ld(n): return pickle.load(open(Z+('/swinsL_val.pkl' if n=='swins' else '/swinbL_val.pkl'),'rb')) if n in ('swins','swinb') else load(n,'val')
def job(n):
    V=ld(n); photos=sorted(V); G=GT(photos); out={}
    D4={ph:d4_objects(V[ph]) for ph in photos}
    for f in (0.8,0.85,0.88,0.9,0.92):
        res={}
        for ph in photos:
            cv,ar,k=paint_labels([c for c in V[ph] if c.meta.get('view',0)==0],f); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e
        out[('single',f)]=res
    for agg,fl in (('mean8',(0.8,0.85,0.88,0.9)),('maxfrac',(0.88,0.9,0.92,0.94))):
        for f in fl:
            res={}
            for ph in photos:
                for o in D4[ph]: o.score=o.meta[agg]
                cv,ar,k=paint_labels(D4[ph],f); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e
            out[(agg,f)]=res
    return n,out
if __name__=='__main__':
    with Pool(4) as p: R=dict(p.map(job,NAMES,chunksize=1))
    pickle.dump(R,open(Z+'/model_by_pop.pkl','wb'))
    d,byimg,PH=load_train(); photos=sorted(R['e40'][('single',0.85)])
    sels={'all':lambda ph,e:True,'target':lambda ph,e:e[:4] in TARGET,'k=3':lambda ph,e:len(PH[ph])==3,'other':lambda ph,e:e[:4] not in TARGET}
    def pq(res,f): return pq_from([c for ph in photos for e,c in zip(PH[ph],res[ph]) if f(ph,e)])['pq']
    print('model   rule          |     all   target     k=3    other')
    for n in NAMES:
        for key in R[n]:
            print('%-7s %-8s %.2f | '%(n,key[0],key[1])+'  '.join('%.4f'%pq(R[n][key],f) for f in sels.values()))
        print()

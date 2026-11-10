"""Label-free check that the all-data test models behave like their hold-out twins: cross-model agreement on val vs test."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
Z='/home/user/work/an/zoo'
def agree(A,B,photos,agg='mean8',thr=0.85):
    na=nb=hit=0; sa=[];sb=[]
    for ph in photos:
        oa=[o for o in d4_objects(A[ph]) if o.meta[agg]>=thr]; ob=[o for o in d4_objects(B.get(ph,[])) if o.meta[agg]>=thr]
        na+=len(oa); nb+=len(ob)
        for o in oa:
            if any(ciou(o,p)>=0.5 for p in ob): hit+=1
    return na,nb,hit
for split,models in [('val',{'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}),
                     ('test',{'e40':load('e40','test'),'swins':pickle.load(open(Z+'/swinsALL_test.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbALL_test.pkl','rb'))})]:
    photos=sorted(models['e40'])
    for a,b in [('e40','swins'),('e40','swinb'),('swins','swinb')]:
        na,nb,hit=agree(models[a],models[b],photos)
        print('%s %s vs %s: objects>=0.85 per photo %.2f / %.2f ; fraction of %s objects matched by %s: %.3f'%(split,a,b,na/len(photos),nb/len(photos),a,b,hit/max(na,1)),flush=True)

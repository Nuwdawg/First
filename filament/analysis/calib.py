"""Label-free calibration check: do the consensus scores have the same distribution on validation
(hold-out twins) and on test (all-data models)? Counts objects per photo above each floor."""
import sys,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from multiprocessing import Pool
Z='/home/user/work/an/zoo'
def scores(args):
    ph,per=args; out={}
    out['e40_single']=[c.score for c in per['e40'] if c.meta['view']==0 and c.area>=100]
    for m in per:
        objs=d4_objects(per[m])
        for agg in ('mean8','maxfrac'):
            out[f'{m}_{agg}']=[o.meta[agg] for o in objs if o.area>=100]
    for agg in ('mean8','maxfrac'):
        out[f'three_{agg}']=[o.score for o in multi_objects(per,agg) if o.area>=100]
    return ph,out
def run(split):
    V={'e40':load('e40',split)}
    V['swins']=pickle.load(open(Z+('/swinsL_val.pkl' if split=='val' else '/swinsALL_test.pkl'),'rb'))
    V['swinb']=pickle.load(open(Z+('/swinbL_val.pkl' if split=='val' else '/swinbALL_test.pkl'),'rb'))
    photos=sorted(V['e40'])
    with Pool(4) as p: res=dict(p.map(scores,[(ph,{m:V[m].get(ph,[]) for m in V}) for ph in photos],chunksize=4))
    return res
if __name__=='__main__':
    R={s:run(s) for s in ('val','test')}
    pickle.dump(R,open(Z+'/calib_scores.pkl','wb'))
    floors=[0.5,0.6,0.7,0.8,0.85,0.88,0.9,0.92,0.94,0.96]
    print('objects per photo with score >= floor (area >= 100), val | test | test/val')
    for key in R['val'][next(iter(R['val']))]:
        print(key)
        for f in floors:
            nv=sum(sum(s>=f for s in R['val'][ph][key]) for ph in R['val'])/len(R['val'])
            nt=sum(sum(s>=f for s in R['test'][ph][key]) for ph in R['test'])/len(R['test'])
            print('   %.2f  %6.2f %6.2f  %.3f'%(f,nv,nt,nt/nv))

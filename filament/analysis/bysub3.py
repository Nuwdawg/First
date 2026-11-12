"""If the test set is triple-annotated photographs, what do the decision rules look like on that subset of validation?
Per-subset PQ (photos with 1 / 2 / 3 annotator records) for floors, consensus rules and geometric transforms."""
import sys,pickle,numpy as np,cv2,collections; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast import Cand
from fast2 import GT, score_photo, paint_labels, H, W
from lib import pq_from, load_train
from multiprocessing import Pool
from geom import tf
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos)
d,byimg,PH=load_train()
K={ph:len(PH[ph]) for ph in photos}
OBJ={}
def prep(ph):
    o={'single':[c for c in M['e40'][ph] if c.meta['view']==0]}
    o['d4']=d4_objects(M['e40'][ph])
    for agg in ('mean8','maxfrac'):
        o['three_'+agg]=multi_objects({m:M[m][ph] for m in M},agg)
    return ph,o
def ev(args):
    name,src,agg,floor,t=args; ent={}
    for ph in photos:
        objs=OBJ[ph][src]
        if agg:
            for o in objs: o.score=o.meta[agg]
        kept=paint_labels(objs,floor)[2]
        if t!=(0,0,None):
            kept=[tf(o,*t) for o in kept]
            cv,ar,k=paint_labels(kept,-1,min_area=1,order=kept)
        else: cv,ar,k=paint_labels(kept,-1,min_area=1,order=kept)
        e,p=score_photo(cv,ar,G.g[ph]); ent[ph]=e
    return name,ent
if __name__=='__main__':
    with Pool(4) as p: OBJ=dict(p.map(prep,photos,chunksize=4))
    jobs=[]
    for f in (0.6,0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.94): jobs.append(('single@%.2f'%f,'single',None,f,(0,0,None)))
    for f in (0.8,0.85,0.88,0.9,0.92,0.94,0.96):
        jobs.append(('three_mean8@%.2f'%f,'three_mean8',None,f,(0,0,None))); jobs.append(('three_maxfrac@%.2f'%f,'three_maxfrac',None,f,(0,0,None)))
    for agg,f in (('mean8',0.85),('maxfrac',0.92)): jobs.append(('e40d4_%s@%.2f'%(agg,f),'d4',agg,f,(0,0,None)))
    T=[(0,0,('dilate','cross',1)),(0,0,('erode','cross',1)),(1,0,None),(-1,0,None),(0,1,None),(0,-1,None)]
    for t in T:
        jobs.append(('single@0.85 %s'%str(t),'single',None,0.85,t)); jobs.append(('three_mean8@0.88 %s'%str(t),'three_mean8',None,0.88,t))
    with Pool(4) as p: R=dict(p.map(ev,jobs,chunksize=1))
    pickle.dump((R,K),open(Z+'/bysub3.pkl','wb'))
    print('photos by records:',collections.Counter(K.values()))
    print('%-46s |   all     k=1     k=2     k=3   | masks/photo k=1 k=2 k=3'%'rule')
    for name,_,_,_,_ in jobs:
        row=[]
        for sel in (None,1,2,3):
            ent=[c for ph in photos if sel is None or K[ph]==sel for c in R[name][ph]]
            row.append(pq_from(ent)['pq'])
        print('%-46s | %.4f  %.4f  %.4f  %.4f'%(name,*row),flush=True)

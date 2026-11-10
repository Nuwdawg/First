import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
names=sys.argv[1].split(',')
V={n:load(n,'val') for n in names}; photos=sorted(V[names[0]]); G=GT(photos)
# two-stage: per-model d4 consensus objects (score = mean over 8 views), then cross-model consensus
PM={}
for n in names:
    CL={ph:cluster(V[n][ph],link=0.5) for ph in photos}
    PM[n]={}
    for ph in photos:
        objs=[consensus(k,8,'vote',0.6) for k in CL[ph]]
        for o in objs: o.meta['src']=n; o.meta['mean8']=o.meta['sum']/8; o.meta['mf']=o.meta['max']*o.meta['n']/8
        PM[n][ph]=objs
for key in ['mean8','mf']:
    for ph in photos:
        for n in names:
            for o in PM[n][ph]: o.score=o.meta[key]
    CL2={ph:cluster([o for n in names for o in PM[n][ph]],link=0.5,min_score=0.01) for ph in photos}
    O={ph:[consensus(k,len(names),'vote',0.5) for k in cl] for ph,cl in CL2.items()}
    evalset(O,G,photos,len(names),'+'.join(names)+' 2stage '+key,FL=(0.6,0.7,0.75,0.8,0.85,0.88,0.9,0.92))

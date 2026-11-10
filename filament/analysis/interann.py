import sys; sys.path.insert(0,'/home/user/work/an')
from lib import *
import itertools
d,byimg,photos=load_train()
pair=collections.defaultdict(list); allc=[]
areas=collections.defaultdict(list)
import time; t0=time.time()
rl={}
def get(e):
    if e not in rl: rl[e]=entry_rles(byimg,e)
    return rl[e]
# pairwise: treat annotator B as "prediction" against annotator A as GT (both directions)
for ph,ents in photos.items():
    if len(ents)<2: continue
    for a,b in itertools.permutations(ents,2):
        iou=iou_matrix(get(a),get(b))
        c=pq_counts(iou); allc.append(c)
        pair[(a.split('-')[0],b.split('-')[0])].append(c)
print('time',time.time()-t0)
print('ALL inter-annotator (B as pred vs A as GT), entry-wise:',pq_from(allc))
print('per ordered pair (gt,pred): n pq sq rq tp fp fn')
for k,v in sorted(pair.items()):
    if len(v)>=8:
        r=pq_from(v); print(k,len(v),'%.3f %.3f %.3f'%(r['pq'],r['sq'],r['rq']),r['tp'],r['fp'],r['fn'])
# objects count & area per annotator within shared triples
for trip in [('010401','010402','010403'),('050101','050102','050103'),('010101','010102','010103'),('020401','020403'),('010101','010102'),('010202','010203')]:
    n=collections.defaultdict(int); ar=collections.defaultdict(float); cnt=0
    for ph,ents in photos.items():
        pf=tuple(sorted(e.split('-')[0] for e in ents))
        if pf!=trip: continue
        cnt+=1
        for e in ents:
            p=e.split('-')[0]; n[p]+=len(byimg[e]); ar[p]+=sum(float(mu.area(r)) for r in get(e))
    print(trip,'photos',cnt,{p:(n[p],round(ar[p]/max(n[p],1)),round(ar[p]/cnt)) for p in trip})

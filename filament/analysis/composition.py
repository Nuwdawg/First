"""What does the leaderboard cluster at 0.55 say about the test set? Score 'copy one annotator's record' on
training photographs with 2 and 3 annotator records, using the official per-record pooled rule."""
import sys,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
from lib import *
d,byimg,photos=load_train()
from pycocotools import mask as mu
res=collections.defaultdict(list); nrec=collections.Counter(); nfil=collections.defaultdict(list)
for ph,ents in photos.items():
    k=len(ents); nrec[k]+=1
    R={e:entry_rles(byimg,e) for e in ents}
    for e in ents: nfil[k].append(len(R[e]))
    if k<2: continue
    for src in ents:               # copy annotator src, score against every record of the photo
        cs=[pq_counts(iou_matrix(R[e],R[src])) for e in ents]
        res[(k,'copy')].append(cs)
        cs2=[pq_counts(iou_matrix(R[e],R[src])) for e in ents if e!=src]
        res[(k,'others')].append(cs2)
print('photos by number of records',dict(nrec))
for k in sorted(nfil): print(' records per photo %d: mean filaments per record %.2f'%(k,np.mean(nfil[k])))
for key in sorted(res):
    flat=[c for cs in res[key] for c in cs]; a=pq_from(flat)
    print(key,'PQ %.4f sq %.3f rq %.3f'%(a['pq'],a['sq'],a['rq']))
# mixtures: fraction f of photos with 3 records, rest 2
import itertools
c3=[c for cs in res[(3,'copy')] for c in cs]; c2=[c for cs in res[(2,'copy')] for c in cs]

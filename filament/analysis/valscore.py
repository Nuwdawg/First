import sys; sys.path.insert(0,'/home/user/work/an')
from lib import *
import json
d,byimg,photos=load_train()
vc=json.load(open('/home/user/work/caches/exp147/val_cache.json'))['photographs']
c0=vc['20110325171014Mh.jpeg']['full'][0]
print({k:(str(v)[:80]) for k,v in c0.items() if k!='embedding'})
def deployed(cands,floor=0.85,min_area=100):
    """paint by descending score; keep only unclaimed pixels; drop <min_area"""
    out=[]; claimed=None
    for c in sorted(cands,key=lambda c:-c['score']):
        if c['score']<floor: continue
        r=c['rle'] if isinstance(c['rle'],dict) else {'size':[H,W],'counts':c['rle']}
        if isinstance(r['counts'],str): r={'size':r['size'],'counts':r['counts'].encode()}
        m=mu.decode(r).astype(bool)
        if claimed is None: claimed=np.zeros((H,W),bool)
        m&=~claimed
        if m.sum()<min_area: continue
        claimed|=m
        out.append(mu.encode(np.asfortranarray(m.astype(np.uint8))))
    return out
preds={}
for fn,v in vc.items():
    preds[fn.split('.')[0]]=deployed(v['full'])
import pickle; pickle.dump(preds,open('/home/user/work/an/val_deployed_preds.pkl','wb'))
allc=[];bypref=collections.defaultdict(list);bymult=collections.defaultdict(list);union=[];bygroup=collections.defaultdict(list)
for ph,pr in preds.items():
    ents=photos[ph]
    gts=[entry_rles(byimg,e) for e in ents]
    for e,g in zip(ents,gts):
        c=pq_counts(iou_matrix(g,pr)); allc.append(c); bypref[e.split('-')[0]].append(c); bymult[len(ents)].append(c); bygroup[e[:4]].append(c)
    allg=[x for g in gts for x in g]
    union.append(greedy_one_to_one(iou_matrix(allg,pr)))
print('ENTRY-WISE (official):',pq_from(allc))
print('n preds',sum(len(p) for p in preds.values()),'records',len(allc))
for k in sorted(bymult): print('multiplicity',k,len(bymult[k]),pq_from(bymult[k]))
for k in sorted(bygroup): print('group',k,len(bygroup[k]),{a:(round(b,3) if isinstance(b,float) else b) for a,b in pq_from(bygroup[k]).items()})
for k in sorted(bypref):
    if len(bypref[k])>=5: print('prefix',k,len(bypref[k]),{a:(round(b,3) if isinstance(b,float) else b) for a,b in pq_from(bypref[k]).items()})
print('UNION one-to-one:',pq_from(union))

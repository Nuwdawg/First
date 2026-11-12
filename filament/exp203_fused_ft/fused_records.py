"""exp203: fused consensus training records for the multi-annotated photographs of the training fold.
For a photograph with two or three annotator records, masks are clustered across annotators (IoU >= 0.5);
a cluster drawn by at least two annotators becomes one target, represented by a real annotator mask
(the most central of three, the smaller of two) - the policy that scored best in analysis/fuse_oracle.py.
Masks drawn by one annotator only are dropped. Validation photographs are never included."""
import sys,json,collections; sys.path.insert(0,'/home/user/work/an')
from lib import *
from fast import Cand
import cluster as CLU
from fuse_oracle import items_of, mk, fuse
d,byimg,photos=load_train()
val=set(k.split('.')[0] for k in json.load(open('/home/user/work/caches/exp147/val_cache.json'))['photographs'])
fname={im['file_name'].split('.')[0]:im['file_name'] for im in d['images']}
train=[ph for ph in sorted(photos) if ph not in val]
assert len(train)==601 and sum(len(photos[ph]) for ph in train)==974, (len(train),)
out=[]; st=collections.Counter()
for ph in train:
    ents=photos[ph]
    if len(ents)<2: continue
    pool=[]
    for e in ents:
        for (x,y,m,a) in items_of(entry_rles(byimg,e)): pool.append(mk(x,y,m,float(a),{'src':e}))
    clusters=CLU.cluster(pool,link=0.5,min_score=0.0)
    multi=[c for c in clusters if len(c.members)>=2]
    st['photos_k%d'%len(ents)]+=1; st['masks_in']+=len(pool); st['clusters_multi']+=len(multi); st['clusters_single']+=len(clusters)-len(multi)
    rles=[]
    for c in sorted(multi,key=lambda c:(-len(c.members),-c.seed.area)):
        o=fuse(c.members,'medoid_small'); full=np.zeros((H,W),np.uint8); full[o.y0:o.y0+o.mask.shape[0],o.x0:o.x0+o.mask.shape[1]]=o.mask
        r=mu.encode(np.asfortranarray(full)); rles.append(dict(size=[H,W],counts=r['counts'].decode('ascii')))
    if rles: out.append(dict(image_id='fused-'+ph,file_name=fname[ph],rles=rles))
    else: st['photos_without_consensus']+=1
json.dump(out,open('/home/user/work/ft/fused_records.json','w'))
print(dict(st)); print('fused records',len(out),'masks',sum(len(r['rles']) for r in out),'mean per record %.2f'%(sum(len(r['rles']) for r in out)/len(out)))

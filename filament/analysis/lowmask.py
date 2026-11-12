"""Masks at a lower probability threshold (logit > -1) reach further along faint filament ends. Do they suit the
annotators who draw bigger filaments? Single-view e40, floor sweep, by annotator population."""
import sys,json,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from fast import Cand
from fast2 import GT, score_photo, paint_labels
from lib import load_train, pq_from
TARGET={'0101','0102','0104','0501'}
d=json.load(open('/home/user/work/caches/exp147/val_cache.json'))['photographs']
V={}
for fn,v in d.items():
    V[fn.split('.')[0]]={k:[Cand(c[k],c['score'],{}) for c in v['full'] if k in c] for k in ('rle','rle_low','rle_high')}
photos=sorted(V); G=GT(photos); dd,byimg,PH=load_train()
print('candidates with low masks:',sum(len(V[p]['rle_low']) for p in photos),'of',sum(len(V[p]['rle']) for p in photos))
sels={'all':lambda ph,e:True,'target':lambda ph,e:e[:4] in TARGET,'k=3':lambda ph,e:len(PH[ph])==3,'other':lambda ph,e:e[:4] not in TARGET,'0104':lambda ph,e:e[:4]=='0104','0501':lambda ph,e:e[:4]=='0501'}
print('masks     floor | area/photo | '+'  '.join('%7s'%s for s in sels))
for k in ('rle','rle_low','rle_high'):
    for f in (0.8,0.85,0.9):
        res={}; A=0
        for ph in photos:
            cv,ar,kk=paint_labels(V[ph][k],f); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e; A+=int(ar.sum())
        print('%-9s %.2f  | %8.0f   | '%(k,f,A/len(photos))+'  '.join('%.4f '%pq_from([c for ph in photos for e,c in zip(PH[ph],res[ph]) if s(ph,e)])['pq'] for s in sels.values()),flush=True)

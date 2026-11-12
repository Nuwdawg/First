"""Minimum mask area by annotator population (the target annotators draw fewer small filaments)."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast2 import GT, score_photo, paint_labels
from lib import load_train, pq_from
from ft_eval import SEL, by_pop
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos)
P={'single@0.85':({ph:[c for c in M['e40'][ph] if c.meta['view']==0] for ph in photos},0.85),
   'three_mean8@0.88':({ph:multi_objects({m:M[m][ph] for m in M},'mean8') for ph in photos},0.88)}
print('%-18s min area masks/ph | '%'rule'+'  '.join('%7s'%s for s in SEL))
for name,(O,f) in P.items():
    for a in (100,150,200,300,400,600):
        res={}; n=0
        for ph in photos:
            cv,ar,k=paint_labels(O[ph],f,min_area=a); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e; n+=len(ar)
        print('%-18s %4d     %5.2f   | '%(name,a,n/len(photos))+'  '.join('%.4f '%x for x in by_pop(res,photos)),flush=True)

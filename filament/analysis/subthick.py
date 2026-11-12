"""Sub-pixel thickness sweep on validation, by annotator population. A binary mask is blurred (Gaussian, sigma 1 px)
and re-thresholded: threshold 0.5 keeps the mask, lower thresholds thicken it by a fraction of a pixel, higher thin it."""
import sys,pickle,numpy as np,cv2,collections; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast import Cand
from fast2 import GT, score_photo, paint_labels, H, W
from lib import pq_from, load_train
from multiprocessing import Pool
Z='/home/user/work/an/zoo'
TARGET={'0101','0102','0104','0501'}
def thick(o,t,sigma=1.0):
    if t==0.5: return o
    pad=4; m=np.pad(o.mask.astype(np.float32),pad)
    b=cv2.GaussianBlur(m,(0,0),sigma)>t
    x0,y0=o.x0-pad,o.y0-pad
    a0=max(x0,0); b0=max(y0,0); a1=min(x0+b.shape[1],W); b1=min(y0+b.shape[0],H)
    b=b[b0-y0:b1-y0,a0-x0:a1-x0]
    n=object.__new__(Cand); n.x0,n.y0,n.mask=a0,b0,b; n.area=int(b.sum()); n.score=o.score; n.meta={}
    return n
if __name__=='__main__':
    M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
    photos=sorted(M['e40']); G=GT(photos); d,byimg,PH=load_train()
    KEPT={'single@0.85':{ph:paint_labels([c for c in M['e40'][ph] if c.meta['view']==0],0.85)[2] for ph in photos},
          'three_mean8@0.88':{ph:rule_multi({m:M[m][ph] for m in M},'mean8',0.88)[2] for ph in photos},
          'three_maxfrac@0.92':{ph:rule_multi({m:M[m][ph] for m in M},'maxfrac',0.92)[2] for ph in photos}}
    def ev(args):
        name,t=args; ent={}; ar0=ar1=0
        for ph in photos:
            objs=[thick(o,t) for o in KEPT[name][ph]]
            ar0+=sum(o.area for o in KEPT[name][ph]); ar1+=sum(o.area for o in objs)
            cv,ar,k=paint_labels(objs,-1,min_area=1,order=objs); e,p=score_photo(cv,ar,G.g[ph]); ent[ph]=e
        return (name,t),(ent,ar1/ar0)
    TS=[0.7,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2]
    jobs=[(n,t) for n in KEPT for t in TS]
    with Pool(4) as p: R=dict(p.map(ev,jobs,chunksize=1))
    pickle.dump(R,open(Z+'/subthick.pkl','wb'))
    sels={'all':lambda ph,e:True,'target':lambda ph,e:e[:4] in TARGET,'other':lambda ph,e:e[:4] not in TARGET,'k=3':lambda ph,e:len(PH[ph])==3,
          '0104':lambda ph,e:e[:4]=='0104','0501':lambda ph,e:e[:4]=='0501','0101':lambda ph,e:e[:4]=='0101','0102':lambda ph,e:e[:4]=='0102'}
    for n in KEPT:
        print(n); print('  thr   area x |  '+'  '.join('%7s'%s for s in sels))
        for t in TS:
            ent,ar=R[(n,t)]
            print('  %.2f  %.3f  |  '%(t,ar)+'  '.join('%.4f '%pq_from([c for ph in photos for e,c in zip(PH[ph],ent[ph]) if f(ph,e)])['pq'] for f in sels.values()),flush=True)

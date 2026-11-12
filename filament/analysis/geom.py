"""Hypothesis 8: the hidden test masks differ geometrically (shift or thickness) from the training masks.
Emulate on validation by transforming the predictions and scoring with the official rule.
Rules replayed: single-view e40 at floors 0.70/0.85/0.90 (public 0.39/0.41/0.40), G (public 0.40), F (public 0.41)."""
import sys,pickle,numpy as np,cv2; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast import Cand
from fast2 import GT, score_photo, paint_labels, H, W
from lib import pq_from
from multiprocessing import Pool
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos)
RULES={}
for f in (0.70,0.85,0.90):
    RULES['e40@%.2f'%f]={ph:paint_labels([c for c in M['e40'][ph] if c.meta['view']==0],f)[2] for ph in photos}
RULES['G']={ph:rule_multi({m:M[m][ph] for m in M},'maxfrac',0.92)[2] for ph in photos}
RULES['F']={ph:rule_multi({m:M[m][ph] for m in M},'mean8',0.88)[2] for ph in photos}
def tf(o,dx,dy,op):
    m=o.mask; x0,y0=o.x0,o.y0
    if op:
        kind,k,it=op; pad=it+1; mm=np.pad(m.astype(np.uint8),pad)
        ker=cv2.getStructuringElement(cv2.MORPH_CROSS if k=='cross' else cv2.MORPH_RECT,(3,3))
        mm=cv2.dilate(mm,ker,iterations=it) if kind=='dilate' else cv2.erode(mm,ker,iterations=it)
        m=mm.astype(bool); x0-=pad; y0-=pad
    x0+=dx; y0+=dy
    a0=max(x0,0); b0=max(y0,0); a1=min(x0+m.shape[1],W); b1=min(y0+m.shape[0],H)
    m=m[b0-y0:b1-y0,a0-x0:a1-x0]
    n=object.__new__(Cand); n.x0,n.y0,n.mask=a0,b0,m; n.area=int(m.sum()); n.score=o.score; n.meta={}
    return n
def evalT(t):
    dx,dy,op=t; out={}
    for r,K in RULES.items():
        ent=[]
        for ph in photos:
            objs=[tf(o,dx,dy,op) for o in K[ph]]
            cv,ar,k=paint_labels(objs,-1,min_area=1,order=objs)
            e,p=score_photo(cv,ar,G.g[ph]); ent+=e
        a=pq_from(ent); out[r]=(a['pq'],a['sq'],a['tp'],a['fp'],a['fn'])
    return t,out
if __name__=='__main__':
    T=[(dx,dy,None) for dy in range(-3,4) for dx in range(-3,4)]
    T+=[(0,0,(kind,k,it)) for kind in ('dilate','erode') for k in ('cross','square') for it in (1,2)]
    with Pool(4) as p: R=dict(p.map(evalT,T,chunksize=2))
    pickle.dump(R,open(Z+'/geom.pkl','wb'))
    names=list(RULES)
    print('transform of predictions            | '+'  '.join('%8s'%n for n in names))
    for t in T:
        print('%-35s | '%str(t)+'  '.join('%8.4f'%R[t][n][0] for n in names),flush=True)

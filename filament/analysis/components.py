"""Predicted masks with several disconnected pieces: keep largest, split, or bridge?"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
import cv2, fast2
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos); base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def comps(mask):
    n,lab,st,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    order=sorted(range(1,n),key=lambda i:-st[i,cv2.CC_STAT_AREA]); return lab,order,st
def mk(o,mask,score=None):
    n=object.__new__(Cand); ys,xs=np.nonzero(mask)
    if len(ys)==0: return None
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    n.x0,n.y0,n.mask=o.x0+x0,o.y0+y0,mask[y0:y1,x0:x1].copy(); n.area=int(n.mask.sum()); n.score=o.score if score is None else score; n.meta={}; return n
def gap(lab,a,b):
    A=(lab==a).astype(np.uint8); d=cv2.distanceTransform(1-A,cv2.DIST_L2,3); return float(d[lab==b].min())
def bridge(mask,lab,order,maxgap):
    """connect each minor piece to the largest piece by a straight band if the gap is at most maxgap"""
    out=mask.copy(); A=(lab==order[0]); ya,xa=np.nonzero(A); w=max(3,int(round(2*A.sum()/max(len(cv2.findContours(A.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0][0]),1))))
    for b in order[1:]:
        yb,xb=np.nonzero(lab==b)
        dA=cv2.distanceTransform(1-A.astype(np.uint8),cv2.DIST_L2,3); i=int(dA[yb,xb].argmin()); g=dA[yb[i],xb[i]]
        if g>maxgap: continue
        pb=(int(xb[i]),int(yb[i])); j=int(((xa-pb[0])**2+(ya-pb[1])**2).argmin()); pa=(int(xa[j]),int(ya[j]))
        tmp=out.astype(np.uint8); cv2.line(tmp,pa,pb,1,thickness=w); out=tmp.astype(bool)
    return out
def transform(objs,mode,minfrac=0.25,mingap=10,maxgap=40):
    res=[];stats=collections.Counter()
    for o in objs:
        lab,order,st=comps(o.mask)
        if len(order)<=1: res.append(o); continue
        stats['multi']+=1
        if mode=='largest':
            n=mk(o,lab==order[0]); res.append(n)
        elif mode=='split':
            big=st[order[0],cv2.CC_STAT_AREA]
            for k,i in enumerate(order):
                a=st[i,cv2.CC_STAT_AREA]
                if k==0 or (a>=minfrac*big and gap(lab,order[0],i)>=mingap):
                    n=mk(o,lab==i,o.score*(1.0 if k==0 else 0.999)); res.append(n)
                # other minor pieces are dropped
        elif mode=='split_keep':
            big=st[order[0],cv2.CC_STAT_AREA]; main=(lab==order[0]).copy()
            for k,i in enumerate(order[1:]):
                a=st[i,cv2.CC_STAT_AREA]
                if a>=minfrac*big and gap(lab,order[0],i)>=mingap: res.append(mk(o,lab==i,o.score*0.999))
                else: main|=(lab==i)
            res.append(mk(o,main))
        elif mode=='bridge':
            res.append(mk(o,bridge(o.mask,lab,order,maxgap)))
    return [r for r in res if r is not None],stats
S={ph:[c for c in M['e40'][ph] if c.meta['view']==0 and c.score>=0.5] for ph in photos}
O3={ph:[o for o in multi_objects({m:M[m][ph] for m in M},'maxfrac') if o.score>=0.5] for ph in photos}
# how big are the minor pieces?
fr=[];gp=[]
for ph in photos:
    for o in S[ph]:
        if o.score<0.85: continue
        lab,order,st=comps(o.mask)
        if len(order)>1: fr.append(1-st[order[0],cv2.CC_STAT_AREA]/o.area); gp.append(gap(lab,order[0],order[1]))
fr=np.array(fr);gp=np.array(gp); print('deployed multi-piece masks on val: %d; share of area outside the largest piece: median %.3f, >10%%: %.2f, >25%%: %.2f; gap to second piece px: median %.1f, quartiles %s'%(len(fr),np.median(fr),(fr>0.1).mean(),(fr>0.25).mean(),np.median(gp),np.quantile(gp,[.25,.75]).round(1)))
for label,objs,floor in [('e40 single @0.85',S,0.85),('three-model maxfrac @0.92',O3,0.92)]:
    r0=run(objs,G,photos,floor); a0=pq_of(r0); print('%-26s as is: PQ %.4f sq %.4f (%d/%d/%d)'%(label,a0['pq'],a0['sq'],a0['tp'],a0['fp'],a0['fn']),flush=True)
    for mode,kw in [('largest',{}),('split',{}),('split_keep',{}),('split_keep',dict(minfrac=0.1,mingap=5)),('split_keep',dict(minfrac=0.5,mingap=20)),('bridge',dict(maxgap=20)),('bridge',dict(maxgap=60)),('bridge',dict(maxgap=1000))]:
        T={}
        for ph in photos: T[ph],_=transform(objs[ph],mode,**kw)
        r=run(T,G,photos,floor); a=pq_of(r); d_,ci,pr=boot(r,r0,photos,n=400)
        print('   %-11s %-28s PQ %.4f sq %.4f (%d/%d/%d) vs as-is %+.4f [%+.4f,%+.4f]'%(mode,str(kw),a['pq'],a['sq'],a['tp'],a['fp'],a['fn'],d_,ci[0],ci[1]),flush=True)

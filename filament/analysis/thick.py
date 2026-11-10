"""Sensitivity of validation PQ to systematic mask thickness changes (for interpreting a possible leaderboard probe)."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import multi_objects
import cv2
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos)
O3={ph:[o for o in multi_objects({m:M[m][ph] for m in M},'maxfrac') if o.score>=0.9] for ph in photos}
def morph(o,op,k):
    pad=3; m=np.pad(o.mask.astype(np.uint8),pad)
    ker=cv2.getStructuringElement(cv2.MORPH_CROSS if k=='cross' else cv2.MORPH_RECT,(3,3))
    m=cv2.dilate(m,ker) if op=='dilate' else cv2.erode(m,ker)
    n=object.__new__(Cand); n.x0,n.y0,n.mask=o.x0-pad,o.y0-pad,m.astype(bool); n.area=int(m.sum()); n.score=o.score; n.meta={}
    # clip to frame
    if n.x0<0 or n.y0<0 or n.x0+n.mask.shape[1]>W or n.y0+n.mask.shape[0]>H:
        x0=max(n.x0,0); y0=max(n.y0,0); x1=min(n.x0+n.mask.shape[1],W); y1=min(n.y0+n.mask.shape[0],H)
        n.mask=n.mask[y0-n.y0:y1-n.y0,x0-n.x0:x1-n.x0]; n.x0,n.y0=x0,y0; n.area=int(n.mask.sum())
    return n
r0=run(O3,G,photos,0.92); a0=pq_of(r0); print('three-model maxfrac @0.92 as is: PQ %.4f sq %.4f (%d/%d/%d); mean area %.0f'%(a0['pq'],a0['sq'],a0['tp'],a0['fp'],a0['fn'],np.mean([o.area for ph in photos for o in O3[ph] if o.score>=0.92])))
for op in ('erode','dilate'):
    for k in ('cross','square'):
        T={ph:[morph(o,op,k) for o in O3[ph]] for ph in photos}
        r=run(T,G,photos,0.92); a=pq_of(r); d_,ci,pr=boot(r,r0,photos,n=300)
        print('  %-6s 1px %-6s: PQ %.4f sq %.4f (%d/%d/%d) %+.4f [%+.4f,%+.4f]; mean area %.0f'%(op,k,a['pq'],a['sq'],a['tp'],a['fp'],a['fn'],d_,ci[0],ci[1],np.mean([o.area for ph in photos for o in T[ph] if o.score>=0.92])),flush=True)
# how thick are annotated filaments? mean width = 2*area/perimeter for matched GT vs predictions

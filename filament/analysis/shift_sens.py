"""Sensitivity of validation PQ to a rigid translation of every predicted mask."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import multi_objects
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos)
O3={ph:[o for o in multi_objects({m:M[m][ph] for m in M},'maxfrac') if o.score>=0.9] for ph in photos}
def shift(o,dx,dy):
    n=object.__new__(Cand); n.mask=o.mask; n.area=o.area; n.score=o.score; n.meta={}
    n.x0=min(max(o.x0+dx,0),W-o.mask.shape[1]); n.y0=min(max(o.y0+dy,0),H-o.mask.shape[0]); return n
r0=run(O3,G,photos,0.92); a0=pq_of(r0); print('as is: PQ %.4f sq %.4f'%(a0['pq'],a0['sq']))
for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,-1),(2,0),(0,2),(2,2),(3,0),(0,3),(4,0)]:
    T={ph:[shift(o,dx,dy) for o in O3[ph]] for ph in photos}
    a=pq_of(run(T,G,photos,0.92)); print('  shift (%+d,%+d): PQ %.4f sq %.4f (%d/%d/%d)  %+.4f'%(dx,dy,a['pq'],a['sq'],a['tp'],a['fp'],a['fn'],a['pq']-a0['pq']),flush=True)

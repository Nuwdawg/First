"""Sub-pixel alignment: is there a systematic offset between predicted masks and rasterised annotations?"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects
import cluster as CLU
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos)
def shifted(o,dx,dy):
    n=object.__new__(Cand); n.mask=o.mask; n.area=o.area; n.score=o.score; n.meta=o.meta
    n.x0=min(max(o.x0+dx,0),W-o.mask.shape[1]); n.y0=min(max(o.y0+dy,0),H-o.mask.shape[0]); return n
# 1) single-view models: integer shift response
for name in ('e40','swins','swinb'):
    S={ph:[c for c in M[name][ph] if c.meta.get('view',0)==0 and c.score>=0.8] for ph in photos}
    a0=pq_of(run(S,G,photos,0.85))['pq']; line='%-6s single view: as is %.4f |'%(name,a0)
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        T={ph:[shifted(o,dx,dy) for o in S[ph]] for ph in photos}; line+=' (%+d,%+d) %+.4f'%(dx,dy,pq_of(run(T,G,photos,0.85))['pq']-a0)
    print(line,flush=True)
# 2) fractional shift through the vote: each member mask votes at its position and at a shifted position
def consensus_frac(k,nsrc,shifts,thr):
    """shifts: list of (dx,dy,weight). Members vote at each shifted position with that weight."""
    ms=k.members; pad=2
    x0=min(m.x0 for m in ms)-pad; y0=min(m.y0 for m in ms)-pad; x1=max(m.x0+m.mask.shape[1] for m in ms)+pad; y1=max(m.y0+m.mask.shape[0] for m in ms)+pad
    acc=np.zeros((y1-y0,x1-x0),np.float32); wsum=0.0
    for m in ms:
        for dx,dy,w in shifts:
            yy=m.y0-y0+dy; xx=m.x0-x0+dx
            acc[yy:yy+m.mask.shape[0],xx:xx+m.mask.shape[1]]+=w*m.mask; wsum+=w
    mask=acc>=thr*wsum-1e-6
    o=object.__new__(Cand); ys,xs=np.nonzero(mask)
    if len(ys)==0: return None
    a,b,c,d=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    o.x0=x0+c; o.y0=y0+a; o.mask=mask[a:b,c:d]; o.area=int(o.mask.sum()); o.meta={'n':len(ms),'max':max(m.score for m in ms),'sum':sum(m.score for m in ms)}; o.score=0
    # clip to frame
    if o.x0<0 or o.y0<0 or o.x0+o.mask.shape[1]>W or o.y0+o.mask.shape[0]>H: return None
    return o
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
# three-model two-stage with fractional shift applied in stage 1 (per-model eight-view vote)
def three(shifts,thr,agg='maxfrac',floor=0.92):
    O={}
    for ph in photos:
        pool=[]
        for name in M:
            for k in CLU.cluster(M[name][ph],link=0.5):
                o=consensus_frac(k,8,shifts,thr)
                if o is None: continue
                o.meta['mean8']=o.meta['sum']/8; o.meta['maxfrac']=o.meta['max']*o.meta['n']/8; o.score=o.meta[agg]; o.meta['src']=name; pool.append(o)
        objs=[CLU.consensus(k,3,'vote',0.5) for k in CLU.cluster(pool,link=0.5,min_score=0.01)]
        for o in objs: o.score=o.meta['sum']/3
        O[ph]=objs
    return run(O,G,photos,floor)
r0=three([(0,0,1.0)],0.6); a0=pq_of(r0); print('three-model maxfrac@0.92, no shift, vote 0.6: PQ %.4f sq %.4f (%d/%d/%d)'%(a0['pq'],a0['sq'],a0['tp'],a0['fp'],a0['fn']),flush=True)
for label,shifts,thr in [('dy -0.5 (vote .5)',[(0,0,1),(0,-1,1)],0.5),('dy -0.5 (vote .6)',[(0,0,1),(0,-1,1)],0.6),('dy -0.33 (vote .5)',[(0,0,2),(0,-1,1)],0.5),('dy -0.33 (vote .6)',[(0,0,2),(0,-1,1)],0.6),
                         ('dy -0.33, dx +0.2 (vote .55)',[(0,0,7),(0,-1,5),(1,0,3)],0.55),('dy -0.25 (vote .55)',[(0,0,3),(0,-1,1)],0.55),('dy +0.33 (vote .55) [control]',[(0,0,2),(0,1,1)],0.55)]:
    r=three(shifts,thr); a=pq_of(r); d_,ci,pr=boot(r,r0,photos,n=400)
    print('  %-30s PQ %.4f sq %.4f (%d/%d/%d) vs no shift %+.4f [%+.4f,%+.4f] P>0 %.2f'%(label,a['pq'],a['sq'],a['tp'],a['fp'],a['fn'],d_,ci[0],ci[1],pr),flush=True)

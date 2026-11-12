"""Label-free check of the eight-view caches: for confident single-view masks, how well does each other view's
matching mask agree (IoU), and is there any systematic offset between them? Validation against test."""
import sys,pickle,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
from zoo import load
from cluster import ciou
from multiprocessing import Pool
def cen(o):
    ys,xs=np.nonzero(o.mask); return o.x0+xs.mean(),o.y0+ys.mean()
def photo(cands):
    base=[c for c in cands if c.meta['view']==0 and c.score>=0.85 and c.area>=100]
    out=[]
    for v in range(1,8):
        others=[c for c in cands if c.meta['view']==v]
        for b in base:
            best=None;bi=0.0
            for o in others:
                # cheap bbox reject is inside ciou
                i=ciou(b,o)
                if i>bi: bi=i;best=o
            if best is None or bi<0.3: out.append((v,0.0,np.nan,np.nan,np.nan,np.nan)); continue
            bx,by=cen(b); ox,oy=cen(best); out.append((v,bi,ox-bx,oy-by,best.area/b.area,best.score-b.score))
    return out
if __name__=='__main__':
    for split in ('val','test'):
        V=load('e40',split)
        with Pool(4) as p: R=[r for L in p.map(photo,[V[ph] for ph in sorted(V)],chunksize=4) for r in L]
        R=np.array(R,float)
        print(split,'photos',len(V),'confident single-view masks x 7 views:',len(R))
        print('  view  found   median IoU  mean IoU   mean dx   mean dy   median area ratio  mean score diff')
        for v in range(1,8):
            r=R[R[:,0]==v]; f=r[:,1]>0
            print('   %d    %.3f    %.4f     %.4f    %+.3f   %+.3f    %.4f             %+.4f'%(v,f.mean(),np.median(r[f,1]),r[f,1].mean(),np.nanmean(r[f,2]),np.nanmean(r[f,3]),np.nanmedian(r[f,4]),np.nanmean(r[f,5])))
        f=R[:,1]>0; print('  all   %.3f    %.4f     %.4f    %+.3f   %+.3f    %.4f             %+.4f'%(f.mean(),np.median(R[f,1]),R[f,1].mean(),np.nanmean(R[f,2]),np.nanmean(R[f,3]),np.nanmedian(R[f,4]),np.nanmean(R[f,5])))

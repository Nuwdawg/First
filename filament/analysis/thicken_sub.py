"""Apply the sub-pixel thickening (Gaussian blur sigma 1 px, re-threshold) to every mask of a submission file.
usage: thicken_sub.py <in.csv> <threshold> <out.csv>   (threshold 0.40 = about 3.5% more area)"""
import sys,csv; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub, check_sub, write_sub
from fast import Cand
from fast2 import paint_labels
from subthick import thick
import numpy as np
src,t,out=sys.argv[1],float(sys.argv[2]),sys.argv[3]
S=read_sub(src); cvs={}; a0=a1=n0=n1=0
for ph,rl in S.items():
    objs=[Cand(r,1.0,{}) for r in rl]; n0+=len(objs); a0+=sum(o.area for o in objs)
    th=[thick(o,t) for o in objs]
    cv,ar,k=paint_labels(th,-1,min_area=100,order=th); cvs[ph]=(cv,len(ar)); n1+=len(ar); a1+=int(ar.sum())
print('rows',write_sub(cvs,out),'from',n0,'area x %.4f'%(a1/a0)); print(check_sub(out))

"""Label-free comparison of validation and test photographs: model confidence and view agreement, image statistics."""
import sys,time,pickle,os; sys.path.insert(0,'/home/user/work/an')
from zoo import *
from cluster import *
import cv2
def stats(split):
    V=load('e40',split); out=[]
    for ph,L in V.items():
        full=[c for c in L if c.meta['view']==0]
        s=np.array(sorted([c.score for c in full],reverse=True))
        cl=cluster(L,link=0.5)
        strong=[k for k in cl if max(m.score for m in k.members)>=0.5]
        agree=np.mean([len(k.members)/8 for k in strong]) if strong else np.nan
        mean8=[sum(m.score for m in k.members)/8 for k in strong]
        out.append((int((s>=0.85).sum()),int((s>=0.5).sum()),s[:5].mean() if len(s)>=5 else np.nan,agree,np.mean([c.area for c in full if c.score>=0.85]) if (s>=0.85).any() else np.nan,int(((s>=0.3)&(s<0.85)).sum()),len(strong),np.mean([m>=0.85 for m in mean8]) if strong else np.nan))
    return np.array(out,float)
for split in ('val','test'):
    A=stats(split)
    print(split,'photos',len(A),'| n>=0.85: %.2f | n>=0.5: %.2f | mean top5 score: %.3f | view agreement: %.3f | mean area: %.0f | n in [0.3,0.85): %.2f | strong clusters %.2f | frac strong clusters with mean8>=0.85: %.3f'%tuple(np.nanmean(A,0)))
    print('   medians:',np.nanmedian(A,0).round(3))
R='/home/user/work/comp/data/MAGFiLO_1.0_Kaggle_2026'
def imstats(path):
    im=cv2.imread(path,cv2.IMREAD_GRAYSCALE).astype(np.float32)
    yy,xx=np.mgrid[:2048,:2048]; disk=((xx-1024)**2+(yy-1024)**2)<(0.8*900)**2
    lap=cv2.Laplacian(im,cv2.CV_32F,ksize=3)
    return im[disk].mean(),im[disk].std(),lap[disk].var()
import random
V=load('e40','val'); T=load('e40','test')
vs=np.array([imstats(f'{R}/train/train_images/{p}.jpeg') for p in sorted(V)])
ts=np.array([imstats(f'{R}/test/test_images/{p}.jpeg') for p in sorted(T)])
print('image stats (disk mean, disk std, Laplacian variance): val mean',vs.mean(0).round(2),'median',np.median(vs,0).round(2))
print('                                                   test mean',ts.mean(0).round(2),'median',np.median(ts,0).round(2))

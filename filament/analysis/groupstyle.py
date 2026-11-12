"""Annotator-group style from the training labels alone (photographs are assigned to groups at random, so
differences between groups are differences between annotators): filaments per record, area, mean width."""
import sys,numpy as np,collections,cv2; sys.path.insert(0,'/home/user/work/an')
from lib import *
from pycocotools import mask as mu
d,byimg,photos=load_train()
TARGET={'0101','0102','0104','0501'}
rows=[]
for ph,ents in photos.items():
    for e in ents:
        A=[];Wd=[]
        for a in byimg[e]:
            p=np.array(a['segmentation'][0],dtype=np.float64).reshape(-1,2)
            x,y=p[:,0],p[:,1]; area=0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))
            per=np.sqrt(((p-np.roll(p,-1,axis=0))**2).sum(1)).sum()
            if per>0 and area>0: A.append(area); Wd.append(2*area/per)
        rows.append((e[:4],e[4:6],len(ents),len(byimg[e]),A,Wd))
def summ(sel,lab):
    R=[r for r in rows if sel(r)]
    if not R: return
    A=np.concatenate([r[4] for r in R]); Wd=np.concatenate([r[5] for r in R])
    print('%-16s records %4d  filaments/record %5.2f  median area %6.0f  median width %.2f  mean width %.2f  area-weighted width %.2f  share<300px %.3f'%(lab,len(R),np.mean([r[3] for r in R]),np.median(A),np.median(Wd),Wd.mean(),(Wd*A).sum()/A.sum(),(A<300).mean()))
summ(lambda r:True,'all records'); summ(lambda r:r[0] in TARGET,'target groups'); summ(lambda r:r[0] not in TARGET,'other groups')
summ(lambda r:r[2]==3,'k=3'); summ(lambda r:r[2]==2,'k=2'); summ(lambda r:r[2]==1,'k=1')
for g in sorted(set(r[0] for r in rows)): summ(lambda r:r[0]==g,'group '+g)
print('within triple-annotated photographs, by annotator:')
for g in sorted(TARGET):
    for a in ('01','02','03'): summ(lambda r:r[0]==g and r[1]==a and r[2]==3,'  %s annotator %s'%(g,a))

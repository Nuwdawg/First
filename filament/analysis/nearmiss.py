import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); G=GT(list(C))
rows=[]   # per GT: best iou, pred area, gt area, inter, n preds touching, n GT touched by that pred
fp_rows=[]
for ph,c in C.items():
    cv,ar,k=paint_labels(c,0.85)
    n=len(ar)
    for e,items in G.g[ph]:
        M=np.zeros((len(items),n)); I=np.zeros((len(items),n))
        for i,(x,y,m,ga) in enumerate(items):
            lab=cv[y:y+m.shape[0],x:x+m.shape[1]][m]; inter=np.bincount(lab,minlength=n+1)[1:]
            I[i]=inter; M[i]=inter/(ga+ar-inter) if n else 0
        for i,(x,y,m,ga) in enumerate(items):
            if n==0 or I[i].max()==0: rows.append((0,0,ga,0,0,0,m.shape[0],m.shape[1])); continue
            j=int(M[i].argmax())
            rows.append((M[i,j],ar[j],ga,I[i,j],int((I[i]>0).sum()),int((I[:,j]>0).sum()),m.shape[0],m.shape[1]))
        for j in range(n):
            if len(items)==0 or I[:,j].max()==0: fp_rows.append((0,ar[j],0,0)); continue
            i=int(M[:,j].argmax()); fp_rows.append((M[i,j],ar[j],items[i][3],I[i,j]))
R=np.array(rows,float); F=np.array(fp_rows,float)
iou=R[:,0]; print('GT total',len(R),'TP',(iou>0.5).sum(),'untouched',(iou==0).sum(),'touched-unmatched',((iou>0)&(iou<=0.5)).sum())
nm=(iou>0)&(iou<=0.5)
print('near-miss IoU hist',np.histogram(iou[nm],bins=[0,.1,.2,.3,.4,.5])[0])
rho=R[nm,1]/R[nm,2]; rec=R[nm,3]/R[nm,2]; prec=R[nm,3]/R[nm,1]
print('near-miss: median pred/GT area ratio %.2f; frac ratio<0.5: %.2f, 0.5-2: %.2f, >2: %.2f'%(np.median(rho),(rho<0.5).mean(),((rho>=0.5)&(rho<=2)).mean(),(rho>2).mean()))
print('near-miss type: pred mostly inside GT (prec>0.7, rec<0.6): %.2f | GT mostly inside pred (rec>0.7,prec<0.6): %.2f | both partial: %.2f'%(((prec>0.7)&(rec<0.6)).mean(),((rec>0.7)&(prec<0.6)).mean(),(~(((prec>0.7)&(rec<0.6))|((rec>0.7)&(prec<0.6)))).mean()))
print('near-miss: GT touched by >1 pred (split): %.2f ; its pred touches >1 GT (merge): %.2f'%((R[nm,4]>1).mean(),(R[nm,5]>1).mean()))
tp=iou>0.5; rho_tp=R[tp,1]/R[tp,2]
print('TP: median area ratio %.3f, mean log ratio %.3f, sd %.3f'%(np.median(rho_tp),np.log(rho_tp).mean(),np.log(rho_tp).std()))
print('by GT area: bin | n | recall(TP) | untouched | near-miss | median ratio(TP) | median ratio (near-miss)')
for a,b in [(0,300),(300,600),(600,1000),(1000,2000),(2000,4000),(4000,1e9)]:
    s=(R[:,2]>=a)&(R[:,2]<b); 
    print('%5d-%-6d %4d  %.2f  %.2f  %.2f  %.2f  %.2f'%(a,min(b,99999),s.sum(),(iou[s]>0.5).mean(),(iou[s]==0).mean(),((iou[s]>0)&(iou[s]<=0.5)).mean(),np.median((R[s&tp,1]/R[s&tp,2])) if (s&tp).any() else 0,np.median(R[s&nm,1]/R[s&nm,2]) if (s&nm).any() else 0))
print('pred-record pairs',len(F),'TP',(F[:,0]>0.5).sum(),'no-touch FP',(F[:,0]==0).sum(),'touch FP',((F[:,0]>0)&(F[:,0]<=0.5)).sum())
print('by pred area: bin | n | precision | no-touch | touch-unmatched')
for a,b in [(100,300),(300,600),(600,1000),(1000,2000),(2000,4000),(4000,1e9)]:
    s=(F[:,1]>=a)&(F[:,1]<b); print('%5d-%-6d %4d  %.2f  %.2f  %.2f'%(a,min(b,99999),s.sum(),(F[s,0]>0.5).mean(),(F[s,0]==0).mean(),((F[s,0]>0)&(F[s,0]<=0.5)).mean()))
pickle.dump((R,F),open('/home/user/work/an/nearmiss.pkl','wb'))

"""On triple-annotated validation photographs: (1) how well does the model's confidence separate objects that at
least two annotators drew from the rest, and what would a perfect selection of the model's own masks score;
(2) how much do two rules' scores scatter against each other on a 90-photograph sample (photo bootstrap)."""
import sys,pickle,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast2 import GT, paint_labels, iou_mat
from lib import load_train, pq_from, pq_counts
Z='/home/user/work/an/zoo'
d,byimg,PH=load_train()
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=[ph for ph in sorted(M['e40']) if len(PH[ph])==3]; G=GT(photos)
def auc(s,y):
    s=np.asarray(s); y=np.asarray(y,bool); r=s.argsort().argsort()+1.0
    return (r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())
def study(label,objs_by_photo,floor):
    rows=[]; ngt=0
    for ph in photos:
        cv,ar,kept=paint_labels(objs_by_photo[ph],0.3); n=len(ar)
        per=np.zeros((3,n))
        for r,(e,items) in enumerate(G.g[ph]):
            ngt+=len(items)
            if n and len(items):
                Mx=iou_mat(cv,ar,items); hit=Mx>0.5; per[r]=(Mx*hit).sum(0)
        for j,o in enumerate(kept): rows.append((o.score,int((per[:,j]>0).sum()),per[:,j].sum()))
    R=np.array(rows); s,nm,V=R[:,0],R[:,1],R[:,2]
    print(label,'objects with score >= 0.3:',len(R),' annotated mask-records:',ngt)
    for lo,hi in ((floor,1.01),(0.7,floor),(0.5,0.7),(0.3,0.5)):
        m=(s>=lo)&(s<hi); c=collections.Counter(nm[m].astype(int))
        print('   score [%.2f,%.2f): %4d objects; matched by 0/1/2/3 annotators: %s; mean value per record %.3f'%(lo,hi,m.sum(),' / '.join('%4.1f%%'%(100*c[k]/max(1,m.sum())) for k in range(4)),V[m].mean()/3 if m.sum() else 0))
    print('   AUC of the score for "at least two annotators": %.3f; for "at least one": %.3f'%(auc(s,nm>=2),auc(s,nm>=1)))
    def pq(mask): return 2*V[mask].sum()/(3*mask.sum()+ngt)
    print('   PQ at the floor %.2f: %.4f'%(floor,pq(s>=floor)))
    best=max(((pq(s>=f),f) for f in np.arange(0.3,0.97,0.01)))
    print('   best single floor: %.4f at %.2f'%best)
    order=np.argsort(-V); cum=np.cumsum(V[order]); k=np.arange(1,len(V)+1); pqs=2*cum/(3*k+ngt); i=pqs.argmax()
    print('   perfect selection of the same masks: %.4f keeping %d objects (%.2f per photo); value of the last kept %.2f'%(pqs[i],i+1,(i+1)/len(photos),V[order][i]))
    print('   keeping exactly the objects matched by >= 2 annotators: %.4f (%d objects); >= 1: %.4f (%d)'%(pq(nm>=2),(nm>=2).sum(),pq(nm>=1),(nm>=1).sum()))
study('deployed single view',{ph:[c for c in M['e40'][ph] if c.meta['view']==0] for ph in photos},0.85)
study('three-model consensus (mean score)',{ph:multi_objects({m:M[m][ph] for m in M},'mean8') for ph in photos},0.88)
# (2) scatter of differences
R3,K=pickle.load(open(Z+'/bysub3.pkl','rb')); P=pickle.load(open(Z+'/pool13_val.pkl','rb')); B=pickle.load(open(Z+'/blend_thick_val.pkl','rb'))
E={'single':R3['single@0.85'],'C (e40 8-view max x frac .92)':R3['e40d4_maxfrac@0.92'],'G (three max x frac .92)':R3['three_maxfrac@0.92'],'F (three mean .88)':R3['three_mean8@0.88'],
   'vote of 8 (>=4)':B[(4,0.5,0.5)][0]}
for k in ('S8noft_k4','five85','d4_swinb','S6noe40_k3','DM_five','V5d4_k3','d4_ftB4'): E[k]={ph:P[ph][k][0] for ph in P}
rng=np.random.default_rng(0); trip=photos
def boot(a,b,n=2000,size=90):
    da=[]
    for _ in range(n):
        s=rng.choice(len(trip),size=size,replace=True)
        ea=[c for i in s for c in E[a][trip[i]]]; eb=[c for i in s for c in E[b][trip[i]]]
        da.append(pq_from(ea)['pq']-pq_from(eb)['pq'])
    return np.mean(da),np.std(da)
print('\ndifference between two rules on a random 90-photograph sample of triple-annotated photographs (mean, sd)')
for a,b in (('C (e40 8-view max x frac .92)','single'),('G (three max x frac .92)','single'),('F (three mean .88)','single'),('vote of 8 (>=4)','single'),('S8noft_k4','vote of 8 (>=4)'),('five85','vote of 8 (>=4)'),('d4_swinb','vote of 8 (>=4)'),('S6noe40_k3','vote of 8 (>=4)'),('DM_five','vote of 8 (>=4)'),('V5d4_k3','vote of 8 (>=4)'),('d4_ftB4','vote of 8 (>=4)')):
    m,s=boot(a,b); print('   %-32s minus %-18s %+.4f  sd %.4f'%(a,b,m,s),flush=True)

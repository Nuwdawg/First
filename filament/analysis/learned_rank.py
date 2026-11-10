"""Cross-fitted ranking of e40 eight-view clusters by predicted expected matched IoU."""
import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects
from ceiling_utils import cand_iou
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
V=load('e40','val'); photos=sorted(V); G=GT(photos)
_,_,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
X=[];Y=[];grp=[];objs=[]
for gi,ph in enumerate(photos):
    cl=cluster(V[ph],link=0.5)
    for k in cl:
        o=consensus(k,8,'vote',0.6)
        if o.area<100 or o.meta['max']<0.2: continue
        sv=np.zeros(8)
        for m in k.members: sv[int(m.meta['view'])]=m.score
        ious=[ciou(o,m) for m in k.members]
        cy=o.y0+o.mask.shape[0]/2; cx=o.x0+o.mask.shape[1]/2; r=np.hypot(cx-1024,cy-1024)/900
        f=[sv.mean(),sv.max(),sv.min(),(sv>0).mean(),sv.std(),np.sort(sv)[2],np.mean([m.meta['m'] for m in k.members]),np.log(o.area),np.mean(ious),np.min(ious),r,
           max(o.mask.shape)/max(min(o.mask.shape),1)]
        y=np.mean([ (lambda v:(v.max() if len(v) and v.max()>0.5 else 0.0))(cand_iou(o,items)) for e,items in G.g[ph]])
        o.meta['mean8']=sv.mean(); o.meta['maxfrac']=sv.max()*(sv>0).mean()
        X.append(f);Y.append(y);grp.append(gi);objs.append((ph,o))
X=np.array(X);Y=np.array(Y);grp=np.array(grp); print('clusters',len(Y),'mean target',Y.mean().round(3),flush=True)
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def evaluate(pred,label,FL):
    O={ph:[] for ph in photos}
    for (ph,o),s in zip(objs,pred): o.score=float(s); O[ph].append(o)
    rows=[]
    for f in FL:
        r=run(O,G,photos,f); rows.append((pq_of(r)['pq'],f,r,pq_from([c for p in photos if kk[p]>=2 for c in r[p]])['pq']))
    b=max(rows,key=lambda t:t[0]); d,ci,pr=boot(b[2],base,photos,n=500)
    print('%-26s best %.4f @%.2f %+.4f [%+.4f,%+.4f] P %.2f | %s | k>=2 best %.4f'%(label,b[0],b[1],d,ci[0],ci[1],pr,' '.join('%.2f:%.4f'%(f,p) for p,f,_,_ in rows),max(r[3] for r in rows)),flush=True)
evaluate(X[:,0],'mean8 (reference)',(0.8,0.85,0.88,0.9))
evaluate(X[:,1]*X[:,3],'maxfrac (reference)',(0.88,0.9,0.92,0.94))
FLm=(0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5)
for name,mk in [('ridge',lambda:Ridge(alpha=1.0)),('gbr d2 n150',lambda:GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=0.05,subsample=0.8,random_state=0)),('gbr d3 n200',lambda:GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=0))]:
    for cols,cl_ in [(list(range(X.shape[1])),'all feats'),([0,1,2,3,4,5],'view scores only')]:
        oof=np.zeros(len(Y))
        for seed in range(3):
            order=np.random.default_rng(seed).permutation(len(photos)); gmap=order[grp]%5
            for fold in range(5):
                tr=gmap!=fold; te=~tr; m=mk(); m.fit(X[tr][:,cols],Y[tr]); oof[te]+=m.predict(X[te][:,cols])/3
        evaluate(oof,name+' '+cl_,FLm)

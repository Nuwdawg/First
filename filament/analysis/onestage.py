"""Single-stage clustering of all 24 views (three models x eight views)."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos); _,_,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
for n in M:
    for ph in photos:
        for c in M[n][ph]: c.meta['src']='%s:%s'%(n,c.meta.get('view',0)); c.meta['model']=n
CL={ph:cluster([c for n in M for c in M[n][ph]],link=0.5) for ph in photos}
def report(label,O,FL):
    rows=[]
    for f in FL:
        r=run(O,G,photos,f); rows.append((pq_of(r)['pq'],f,r,pq_from([c for p in photos if kk[p]>=2 for c in r[p]])['pq']))
    b=max(rows,key=lambda t:t[0]); d_,ci,pr=boot(b[2],base,photos,n=500)
    print('%-30s best %.4f @%.2f %+.4f [%+.4f,%+.4f] | %s | k>=2 best %.4f'%(label,b[0],b[1],d_,ci[0],ci[1],' '.join('%.2f:%.4f'%(f,p) for p,f,_,_ in rows),max(r[3] for r in rows)),flush=True)
O={ph:[consensus(k,24,'vote',0.6) for k in cl] for ph,cl in CL.items()}
report('24 views, mean24',O,(0.75,0.8,0.85,0.88,0.9))
for ph in photos:
    for o in O[ph]: o.score=o.meta['max']*o.meta['n']/24
report('24 views, max x frac24',O,(0.85,0.88,0.9,0.92,0.94,0.96))
for ph in photos:
    for o,k in zip(O[ph],CL[ph]):
        per={}
        for m in k.members: per.setdefault(m.meta['model'],[]).append(m.score)
        o.score=np.mean([max(per.get(n,[0]))*len(per.get(n,[]))/8 for n in M])
report('24 views, mean of per-model maxfrac',O,(0.85,0.88,0.9,0.92,0.94))

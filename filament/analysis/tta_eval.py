"""TTA consensus per model on validation: d4 only, and for e40 also scale/tiles."""
import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from zoo_eval import *
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def evalset(O,G,photos,nsrc,label,FL=(0.75,0.8,0.85,0.88,0.9,0.92,0.94)):
    out=[]
    for agg in ['mean','max*frac']:
        for ph in photos:
            for o in O[ph]: o.score=o.meta['sum']/nsrc if agg=='mean' else o.meta['max']*o.meta['n']/nsrc
        rows=[]
        for f in FL:
            r=run(O,G,photos,f); a=pq_of(r); rows.append((a['pq'],f,r,a))
        best=max(rows,key=lambda t:t[0]); d,ci,pr=boot(best[2],base,photos,n=600)
        print('%-22s %-8s best %.4f @%.2f (%d/%d/%d) vs deployed %+.4f [%+.4f,%+.4f] P>0 %.2f | %s'%(label,agg,best[0],best[1],best[3]['tp'],best[3]['fp'],best[3]['fn'],d,ci[0],ci[1],pr,' '.join('%.2f:%.4f'%(f,p) for p,f,_,_ in rows)),flush=True)
        out.append(best)
    return out
if __name__=='__main__':
    for n in sys.argv[1].split(','):
        V=load(n,'val'); photos=sorted(V); G=GT(photos)
        S={ph:[c for c in V[ph] if c.meta['view']==0] for ph in photos}
        best1=max((pq_of(run(S,G,photos,f))['pq'],f) for f in (0.8,0.85,0.88,0.9,0.92))
        print('== %s single view best %.4f @%.2f'%(n,best1[0],best1[1]),flush=True)
        CL={ph:cluster(V[ph],link=0.5) for ph in photos}
        O={ph:[consensus(k,8,'vote',0.6) for k in cl] for ph,cl in CL.items()}
        evalset(O,G,photos,8,n+' d4 vote0.6')

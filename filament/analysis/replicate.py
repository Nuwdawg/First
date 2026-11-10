import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects
d,byimg,PH=load_train()
for n in sys.argv[1].split(','):
    V=load(n,'val'); photos=sorted(V); G=GT(photos); kk={ph:len(PH[ph]) for ph in photos}
    S={ph:[c for c in V[ph] if c.meta['view']==0] for ph in photos}
    smax=max(c.score for ph in photos for c in S[ph]); FLs=[smax*q for q in (0.6,0.7,0.75,0.8,0.85,0.88,0.9,0.92,0.94,0.96)]
    def sweep(O):
        rows=[]
        for f in FLs:
            r=run(O,G,photos,f); a=pq_of(r)['pq']; m=pq_from([c for ph in photos if kk[ph]>=2 for c in r[ph]])['pq']; rows.append((a,m,f))
        return rows
    rs=sweep(S); b=max(rs); print('%-7s single  best all %.4f @%.3f | best k>=2 %.4f'%(n,b[0],b[2],max(r[1] for r in rs)),flush=True)
    O={ph:d4_objects(V[ph]) for ph in photos}
    for agg in ('mean8','maxfrac'):
        for ph in photos:
            for o in O[ph]: o.score=o.meta[agg]
        rd=sweep(O); b2=max(rd); print('%-7s d4 %-7s best all %.4f @%.3f (%+.4f) | best k>=2 %.4f (%+.4f)'%(n,agg,b2[0],b2[2],b2[0]-b[0],max(r[1] for r in rd),max(r[1] for r in rd)-max(r[1] for r in rs)),flush=True)

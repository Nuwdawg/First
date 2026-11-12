"""Which masks should the voted objects carry: pixel majority over the files, or the three-model consensus mask where one exists?"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from blend_val import *
T3={ph:multi_objects({m:M[m][ph] for m in ('e40','swins','swinb')},'mean8') for ph in photos}
def run_rule(names,need,masksrc,vote=0.5):
    res={};rows=0;rep=0
    for ph in photos:
        pool=[]
        for n in names:
            for c in files[n][ph]: c.meta['src']=n; pool.append(c)
        objs=[]
        for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(names),'vote',vote)
            if o.meta['n']<need: continue
            o.score=o.meta['n']
            if masksrc=='three':
                best=None;bi=0.5
                for t in T3[ph]:
                    v=CLU.ciou(o,t)
                    if v>=bi: bi=v;best=t
                if best is not None:
                    n=object.__new__(Cand); n.x0,n.y0,n.mask,n.area,n.score,n.meta=best.x0,best.y0,best.mask,best.area,o.score,{}; o=n; rep+=1
            objs.append(o)
        cv,ar,k=paint_labels(objs,-1,order=sorted(objs,key=lambda o:-o.score)); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[ph]=e; rows+=len(ar)
    return res,rows,rep
for label,ns,need in [('8 mixed files, >=4',['deployed','e40_d4_mean','three_mean','swins_d4','swinb_d4','ctrl31_d4','ctrl7','head'],4),('6 single-view files, >=3',['deployed','swins_1','swinb_1','ctrl7','head','focal'],3)]:
    r0,rows,_=run_rule(ns,need,'vote'); a0=pq_of(r0)
    for v in (0.4,0.6):
        rv,_,_=run_rule(ns,need,'vote',vote=v); av=pq_of(rv); print('%s | pixel vote threshold %.1f: PQ %.4f sq %.4f'%(label,v,av['pq'],av['sq']),flush=True)
    r1,rows1,rep=run_rule(ns,need,'three'); a1=pq_of(r1); d_,ci,pr=boot(r1,r0,photos,n=300)
    print('%s | file-majority masks: PQ %.4f sq %.4f | three-model masks (%d of %d replaced): PQ %.4f sq %.4f  %+.4f [%+.4f,%+.4f]'%(label,a0['pq'],a0['sq'],rep,rows1,a1['pq'],a1['sq'],d_,ci[0],ci[1]),flush=True)

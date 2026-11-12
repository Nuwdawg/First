"""Borderline objects in the file-level vote decided by three-model consensus confidence (validation emulation)."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from blend_val import *      # builds the fourteen validation 'files' and helpers
three=multi_objects  # alias
T3={ph:multi_objects({m:M[m][ph] for m in ('e40','swins','swinb')},'mean8') for ph in photos}
def run_rule(names,rule):
    res={};rows=0
    for ph in photos:
        pool=[]
        for n in names:
            for c in files[n][ph]: c.meta['src']=n; pool.append(c)
        objs=[]
        for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(names),'vote',0.5); n=o.meta['n']
            s=max([t.score for t in T3[ph] if CLU.ciou(o,t)>=0.5] or [0.0])
            keep,rank=rule(n,s,len(names))
            if keep: o.score=rank; objs.append(o)
        cv,ar,k=paint_labels(objs,-1,order=sorted(objs,key=lambda o:-o.score)); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[ph]=e; rows+=len(ar)
    return res,rows
names=['deployed','e40_d4_mean','three_mean','swins_d4','swinb_d4','ctrl31_d4','ctrl7','head']
names2=['deployed','swins_1','swinb_1','ctrl7','head','focal']     # no consensus files among the members
for label,ns in [('8 mixed files',names),('6 single-view files',names2)]:
    N=len(ns); half=N//2
    r0,rows0=run_rule(ns,lambda n,s,N:(n>=half,n+s)); a0=pq_of(r0)['pq']; print('%s | pure vote >=%d: %.4f (%d masks)'%(label,half,a0,rows0),flush=True)
    for lo,hi,t in [(half-1,half,0.85),(half-1,half,0.9),(half-1,half+1,0.85),(half-1,half+1,0.9),(half-2,half,0.9),(half-2,half+1,0.88),(1,half+1,0.9),(1,N,0.85),(1,N,0.88)]:
        r,rows=run_rule(ns,lambda n,s,N,lo=lo,hi=hi,t=t:((n>hi) or (n>=lo and s>=t),n+s)); d_,ci,pr=boot(r,r0,photos,n=300)
        print('   keep if votes > %d, or votes >= %d and confidence >= %.2f: %.4f (%d masks)  vs pure vote %+.4f [%+.4f,%+.4f]'%(hi,lo,t,pq_of(r)['pq'],rows,d_,ci[0],ci[1]),flush=True)

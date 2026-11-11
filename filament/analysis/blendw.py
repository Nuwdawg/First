"""Weighted consensus of distinct earlier submissions; weight by the public score each file displayed."""
import sys,glob,os,csv; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub, check_sub
from lib import *
from fast import Cand
import cluster as CLU
from fast2 import paint_labels, canvas_to_rles
S='/home/user/work/subs/'
def load_tier(d,w): return {os.path.basename(f)[:-4]:(read_sub(f),w) for f in sorted(glob.glob(S+d+'/*.raw'))}
def distinct(pool,order,thr=0.97):
    photos=sorted(set().union(*[set(s) for s,_ in pool.values()])); kept=[]
    def agree(a,b): return pq_from([pq_counts(iou_matrix(pool[a][0].get(ph,[]),pool[b][0].get(ph,[]))) for ph in photos])['pq']
    for n in order:
        if all(agree(n,k)<thr for k in kept): kept.append(n)
    return kept,photos
def build(pool,kept,photos,thr,out,vote=0.5):
    W=sum(pool[k][1] for k in kept); cvs={}; n=0
    for ph in photos:
        cands=[]
        for k in kept:
            for r in pool[k][0].get(ph,[]): cands.append(Cand(r,pool[k][1],{'src':k}))
        objs=[]
        for cl in CLU.cluster(cands,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(kept),'wvote',vote); o.score=o.meta['sum']/W; objs.append(o)
        cv,ar,kk=paint_labels(objs,thr-1e-9,min_area=100); cvs[ph]=(cv,len(ar)); n+=len(ar)
    with open(out,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(cvs):
            cv,m=cvs[ph]
            for i,r in enumerate(canvas_to_rles(cv,m)): w.writerow([f'{ph}_{i+1}',r['counts'].decode()])
    print(os.path.basename(out),'members',len(kept),'weight sum %.1f'%W,'threshold %.2f -> rows %d'%(thr,n),check_sub(out),flush=True)
if __name__=='__main__':
    pool={}; pool.update(load_tier('lb41',1.0)); pool.update(load_tier('lb40',0.6)); pool.update(load_tier('lb39',0.3))
    order=sorted(pool,key=lambda n:(-pool[n][1],-int(n)))      # best tier first, newest first within a tier
    kept,photos=distinct(pool,order)
    print('distinct:',[(k,pool[k][1]) for k in kept])
    build(pool,kept,photos,0.5,S+'blendW_thr50.csv'); build(pool,kept,photos,0.4,S+'blendW_thr40.csv')
    k8=[k for k in kept if pool[k][1]==1.0]; build(pool,k8,photos,0.375,S+'blend8_thr38.csv'); build(pool,k8,photos,0.5,S+'blend8_check50.csv')

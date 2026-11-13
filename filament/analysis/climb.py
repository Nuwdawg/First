"""Object table of the eight-file vote on test: for every cluster of masks across the eight distinct 0.41 submissions,
the number of files that contain it (votes) and the deployed model's single-view confidence of the matching candidate.
Used to write variants of the vote file that differ in one well-defined group of marginal objects."""
import sys,os,csv,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub, check_sub
from lib import *
from fast import Cand
import cluster as CLU
from fast2 import paint_labels, canvas_to_rles
from zoo import load
S='/home/user/work/subs/'; PK='/home/user/work/an/zoo/climb_table.pkl'
K8=['58219271','58186186','58160222','58159470','58127006','57816701','56648050','56309562']
def table():
    if os.path.exists(PK): return pickle.load(open(PK,'rb'))
    pool={k:read_sub(S+f'lb41/{k}.raw') for k in K8}; E=load('e40','test')
    photos=sorted(set().union(*[set(s) for s in pool.values()])); T={}
    for ph in photos:
        cands=[]
        for k in K8:
            for r in pool[k].get(ph,[]): cands.append(Cand(r,1.0,{'src':k}))
        objs=[]
        single=[c for c in E.get(ph,[]) if c.meta['view']==0 and c.score>=0.05]
        for cl in CLU.cluster(cands,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(K8),'wvote',0.5); o.score=o.meta['sum']/len(K8); v=int(round(o.meta['sum']))
            best=0.0
            for c in single:
                if c.score>best and CLU.ciou(o,c)>=0.5: best=c.score
            o.meta={'v':v,'e40':best,'src':'vote'}; objs.append(o)
        T[ph]=objs
    pickle.dump(T,open(PK,'wb'),protocol=4); return T
def write(T,keep,out):
    rows=0
    with open(out,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(T):
            sel=[o for o in T[ph] if keep(o)]
            cv,ar,k=paint_labels(sel,-1,min_area=100)
            for i,r in enumerate(canvas_to_rles(cv,len(ar))): w.writerow([f'{ph}_{i+1}',r['counts'].decode()]); rows+=1
    return rows
if __name__=='__main__':
    T=table(); import collections
    allo=[o for ph in T for o in T[ph] if o.area>=100]
    c=collections.Counter(o.meta['v'] for o in allo); print('objects by votes:',dict(sorted(c.items())))
    med={}
    for v in range(1,9):
        e=np.array([o.meta['e40'] for o in allo if o.meta['v']==v]); med[v]=float(np.median(e)) if len(e) else 0
        print('  votes %d: %4d objects; e40 confidence quartiles %s; share without an e40 candidate %.2f'%(v,len(e),np.round(np.percentile(e,[25,50,75]),3) if len(e) else '-',(e==0).mean() if len(e) else 0))
    r=write(T,lambda o:o.meta['v']>=4,S+'climb/check_thr50.csv') if os.makedirs(S+'climb',exist_ok=True) is None else 0
    from build_sub import read_sub as rs
    a=rs(S+'climb/check_thr50.csv'); b=rs(S+'blend41_thr50.csv')
    print('reproduction of blend41_thr50: rows',r,'agreement %.4f'%pq_from([pq_counts(iou_matrix(a.get(ph,[]),b.get(ph,[]))) for ph in sorted(set(a)|set(b))])['pq'])
    V={ 'c1_v5_plus_v4hi': lambda o:o.meta['v']>=5 or (o.meta['v']==4 and o.meta['e40']>=med[4]),
        'c2_v5_plus_v4lo': lambda o:o.meta['v']>=5 or (o.meta['v']==4 and o.meta['e40']<med[4]),
        'c3_v4_plus_v3hi': lambda o:o.meta['v']>=4 or (o.meta['v']==3 and o.meta['e40']>=med[3]),
        'c4_v4_plus_v3lo': lambda o:o.meta['v']>=4 or (o.meta['v']==3 and o.meta['e40']<med[3]),
        'c5_v4_minus_v5lo': lambda o:o.meta['v']>=6 or (o.meta['v']==5 and o.meta['e40']>=med[5]) or o.meta['v']==4,
        'c6_v4_minus_v5hi': lambda o:o.meta['v']>=6 or (o.meta['v']==5 and o.meta['e40']<med[5]) or o.meta['v']==4,
        'c7_v6': lambda o:o.meta['v']>=6,
        'c8_v4_plus_v2hi': lambda o:o.meta['v']>=4 or (o.meta['v']==2 and o.meta['e40']>=0.85),
        'c9_v4_plus_v1hi': lambda o:o.meta['v']>=4 or (o.meta['v']==1 and o.meta['e40']>=0.85)}
    for n,k in V.items():
        r=write(T,k,S+f'climb/{n}.csv'); print(n,'rows',r,check_sub(S+f'climb/{n}.csv'),flush=True)

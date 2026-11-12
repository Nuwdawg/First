"""Candidates for 13 November, one code path for the validation replay and the test files.
S9: vote of nine single-view models (majority of files, pixel-majority masks).
DM: the deployed selection (e40 single view, floor 0.85) with three-model consensus masks."""
import sys,pickle,json,numpy as np; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast import Cand
from fast2 import GT, score_photo, paint_labels
from lib import load_train, pq_from
import cluster as CLU
from ft_eval import SEL, by_pop, load_simple, load_zoo
from maskswap import swap
from build_sub import check_sub
from multiprocessing import Pool
Z='/home/user/work/an/zoo'; X='/home/user/work/caches/exp202'; S='/home/user/work/subs/'
FLOOR={'e40':0.85,'ctrl31':0.88,'ctrl7':0.90,'focal':0.90,'head':0.90,'swins':0.90,'swinb':0.90,'ftA6':0.85,'ftB4':0.90}
def sources(split):
    M={n:load(n,split) for n in ('e40','ctrl31','ctrl7','focal','head')}
    M['swins']=pickle.load(open(Z+('/swinsL_val.pkl' if split=='val' else '/swinsALL_test.pkl'),'rb'))
    M['swinb']=pickle.load(open(Z+('/swinbL_val.pkl' if split=='val' else '/swinbALL_test.pkl'),'rb'))
    M['ftA6']=load_simple(X+'/ftA/e006_cache.json') if split=='val' else load_zoo(X+'/test_ftA_e006/test_cache.json')
    M['ftB4']=load_zoo(X+('/d4_best/val_cache.json' if split=='val' else '/d4_best/test_cache.json'))
    return M
def as_file(cands,floor):
    cv,ar,kept=paint_labels([c for c in cands if c.meta.get('view',0)==0],floor); L=[]
    for i in range(len(ar)):
        m=(cv==i+1); ys,xs=np.nonzero(m); c=object.__new__(Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,m[y0:y1,x0:x1].copy(),int(ar[i]),1.0,{}; L.append(c)
    return L
def vote(files,k):
    pool=[]
    for n,L in files.items():
        for c in L: c.meta['src']=n; pool.append(c)
    objs=[]
    for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
        o=CLU.consensus(cl,len(files),'vote',0.5); o.score=o.meta['n']/len(files); objs.append(o)
    return paint_labels(objs,k/len(files)-1e-9)
CUR={}          # per-split sources and ground truth, set before the pool forks
def photo(ph):
    per={n:CUR['M'][n].get(ph,[]) for n in CUR['M']}; out={}
    files={n:as_file(per.get(n,[]),FLOOR[n]) for n in FLOOR}
    res={}
    for k in (4,5,6):
        cv,ar,kept=vote(files,k); res['S9_k%d'%k]=(cv,ar)
    files7={n:files[n] for n in ('e40','ctrl31','ctrl7','focal','head','swins','swinb')}
    cv,ar,kept=vote(files7,4); res['S7_k4']=(cv,ar)
    single=[c for c in per['e40'] if c.meta.get('view',0)==0]
    dep=paint_labels(single,0.85)[2]
    three=multi_objects({m:per[m] for m in ('e40','swins','swinb')},'mean8')
    L=swap(dep,three,0.3)[0]; cv,ar,kept=paint_labels(L,-1,min_area=100,order=L); res['DM']=(cv,ar)
    for k,(cv,ar) in res.items():
        if CUR['G'] is not None: out[k]=(score_photo(cv,ar,CUR['G'].g[ph])[0],len(ar))
        else: out[k]=[r['counts'].decode() for r in canvas_to_rles(cv,len(ar))]
    return ph,out
if __name__=='__main__':
    import csv
    from fast2 import canvas_to_rles
    d,byimg,PH=load_train()
    CUR['M']=sources('val'); photos=sorted(CUR['M']['e40']); CUR['G']=GT(photos)
    with Pool(3) as p: R=dict(p.imap_unordered(photo,photos,chunksize=1))
    print('validation replay             masks/ph | '+'  '.join('%7s'%s for s in SEL))
    for k in R[photos[0]]:
        res={ph:R[ph][k][0] for ph in photos}; n=sum(R[ph][k][1] for ph in photos)
        print('%-28s %5.2f    | '%(k,n/len(photos))+'  '.join('%.4f '%x for x in by_pop(res,photos)),flush=True)
    del R; CUR['M']=None; import gc; gc.collect()
    CUR['M']=sources('test'); CUR['G']=None; tp=sorted(CUR['M']['e40'])
    with Pool(3) as p: RT=dict(p.imap_unordered(photo,tp,chunksize=1))
    for k,name in (('S9_k5','voteS9_k5.csv'),('S9_k4','voteS9_k4.csv'),('S7_k4','voteS7_k4.csv'),('DM','deployed_sel_three_masks.csv')):
        with open(S+name,'w',newline='') as f:
            w=csv.writer(f); w.writerow(['filament_id','segmentation_rle']); rows=0
            for ph in tp:
                for i,c in enumerate(RT[ph][k]): w.writerow([f'{ph}_{i+1}',c]); rows+=1
        print(name,'rows',rows,check_sub(S+name),flush=True)

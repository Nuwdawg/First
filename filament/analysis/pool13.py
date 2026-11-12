"""A pool of strong, mutually different prediction rules, each replayed on validation (by annotator population)
and written for test with the same code path. Families: votes of single-view models, eight-view consensus per
model, votes of eight-view files, multi-model consensus, deployed selection with consensus masks."""
import sys,pickle,csv,os,numpy as np; sys.path.insert(0,'/home/user/work/an')
from decode import *
from fast2 import GT, score_photo, paint_labels, canvas_to_rles
from lib import load_train
from ft_eval import SEL, by_pop
from maskswap import swap
from build_sub import check_sub
import tickets13 as T
from multiprocessing import Pool
S='/home/user/work/subs/pool13/'; os.makedirs(S,exist_ok=True)
D4M=('e40','swins','swinb','ctrl31','ftB4')
SV={'S5a_k3':(('e40','swins','swinb','ctrl31','ftB4'),3),'S5b_k3':(('e40','swinb','ctrl7','head','ftA6'),3),
    'S6noe40_k3':(('swins','swinb','ctrl31','ctrl7','focal','head'),3),'S8noft_k4':(('e40','swins','swinb','ctrl31','ctrl7','focal','head','ftA6'),4)}
MM={'three85':(('e40','swins','swinb'),0.85),'e40swinb85':(('e40','swinb'),0.85),'swinsswinb85':(('swins','swinb'),0.85),'ftswinsswinb85':(('ftB4','swins','swinb'),0.85),
    'five85':(D4M,0.85),'five88':(D4M,0.88),'five80':(D4M,0.80),'e40fam85':(('e40','ctrl31','ftB4'),0.85)}
D4F={'e40':0.85,'swins':0.88,'swinb':0.88,'ctrl31':0.85,'ftB4':0.85}
CUR={}
def photo(ph):
    per={n:CUR['M'][n].get(ph,[]) for n in CUR['M']}; res={}
    files={n:T.as_file(per[n],T.FLOOR[n]) for n in T.FLOOR}
    for name,(mem,k) in SV.items():
        cv,ar,kept=T.vote({n:files[n] for n in mem},k); res[name]=(cv,ar)
    d4={m:d4_objects(per[m]) for m in D4M}; d4files={}
    for m in D4M:
        for o in d4[m]: o.score=o.meta['mean8']
        cv,ar,kept=paint_labels(d4[m],D4F[m]); res['d4_'+m]=(cv,ar)
        d4files[m]=T.as_file([],0) if len(ar)==0 else None
        L=[]
        for i in range(len(ar)):
            mk=(cv==i+1); ys,xs=np.nonzero(mk); c=object.__new__(T.Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
            c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,mk[y0:y1,x0:x1].copy(),int(ar[i]),1.0,{}; L.append(c)
        d4files[m]=L
    cv,ar,kept=T.vote(d4files,3); res['V5d4_k3']=(cv,ar)
    cv,ar,kept=T.vote(d4files,2); res['V5d4_k2']=(cv,ar)
    mix=dict(d4files); mix.update({'s_'+n:files[n] for n in ('e40','ctrl7','head','focal','ftA6')})
    cv,ar,kept=T.vote(mix,5); res['V10mix_k5']=(cv,ar)
    objs={}
    for name,(mem,f) in MM.items():
        key=tuple(mem)
        if key not in objs:
            pool=[]
            for m in mem:
                for o in d4[m]:
                    o.score=o.meta['mean8']; o.meta['src']=m; pool.append(o)
            n=len(mem); O=[consensus(k,n,'vote',0.5) for k in cluster(pool,link=0.5,min_score=0.01)]
            for o in O: o.score=o.meta['sum']/n
            objs[key]=O
        cv,ar,kept=paint_labels(objs[key],f); res[name]=(cv,ar)
    dep=paint_labels([c for c in per['e40'] if c.meta.get('view',0)==0],0.85)[2]
    L=swap(dep,objs[tuple(D4M)],0.3)[0]; cv,ar,kept=paint_labels(L,-1,min_area=100,order=L); res['DM_five']=(cv,ar)
    out={}
    for k,(cv,ar) in res.items():
        if CUR['G'] is not None: out[k]=(score_photo(cv,ar,CUR['G'].g[ph])[0],len(ar))
        else: out[k]=[r['counts'].decode() for r in canvas_to_rles(cv,len(ar))]
    return ph,out
if __name__=='__main__':
    d,byimg,PH=load_train()
    CUR['M']=T.sources('val'); photos=sorted(CUR['M']['e40']); CUR['G']=GT(photos)
    with Pool(3) as p: R=dict(p.imap_unordered(photo,photos,chunksize=1))
    print('validation replay             masks/ph | '+'  '.join('%7s'%s for s in SEL),flush=True)
    for k in R[photos[0]]:
        res={ph:R[ph][k][0] for ph in photos}; n=sum(R[ph][k][1] for ph in photos)
        print('%-28s %5.2f    | '%(k,n/len(photos))+'  '.join('%.4f '%x for x in by_pop(res,photos)),flush=True)
    pickle.dump(R,open('/home/user/work/an/zoo/pool13_val.pkl','wb'))
    del R; CUR['M']=None; import gc; gc.collect()
    CUR['M']=T.sources('test'); CUR['G']=None; tp=sorted(CUR['M']['e40'])
    with Pool(3) as p: RT=dict(p.imap_unordered(photo,tp,chunksize=1))
    for k in RT[tp[0]]:
        with open(S+k+'.csv','w',newline='') as f:
            w=csv.writer(f); w.writerow(['filament_id','segmentation_rle']); rows=0
            for ph in tp:
                for i,c in enumerate(RT[ph][k]): w.writerow([f'{ph}_{i+1}',c]); rows+=1
        print(k,'rows',rows,check_sub(S+k+'.csv'),flush=True)

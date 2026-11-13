"""Does the deployed model's confidence separate good from bad objects inside a vote group? Validation replay of the
eight-file vote: for each vote count, the effect on PQ of the high- and low-confidence halves (added or removed
relative to the 'at least 4 of 8' file), on the whole validation set and with a 90-photograph bootstrap."""
import sys,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
import cluster as CLU, fast2
from lib import load_train, pq_from
from multiprocessing import Pool
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb')),'ctrl31':load('ctrl31','val'),'ctrl7':load('ctrl7','val'),'head':load('head','val')}
photos=sorted(M['e40']); G=GT(photos)
def paint_file(objs,floor):
    cv,ar,kept=paint_labels(objs,floor); L=[]
    for i in range(len(ar)):
        m=(cv==i+1); ys,xs=np.nonzero(m); c=object.__new__(Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,m[y0:y1,x0:x1].copy(),int(ar[i]),1.0,{}; L.append(c)
    return L
def sc(objs,agg):
    for o in objs: o.score=o.meta[agg]
    return objs
def photo(ph):
    single=[c for c in M['e40'][ph] if c.meta.get('view',0)==0]
    files={'deployed':paint_file(single,0.85),'e40_d4_mean':paint_file(sc(d4_objects(M['e40'][ph]),'mean8'),0.85),'three_mean':paint_file(multi_objects({m:M[m][ph] for m in ('e40','swins','swinb')},'mean8'),0.88),
           'swins_d4':paint_file(sc(d4_objects(M['swins'][ph]),'mean8'),0.88),'swinb_d4':paint_file(sc(d4_objects(M['swinb'][ph]),'mean8'),0.88),
           'ctrl31_d4':paint_file(sc(d4_objects(M['ctrl31'][ph]),'mean8'),0.85),'ctrl7':paint_file([c for c in M['ctrl7'][ph] if c.meta.get('view',0)==0],0.88),
           'head':paint_file([c for c in M['head'][ph] if c.meta.get('view',0)==0],0.90)}
    pool=[]
    for n,L in files.items():
        for c in L: c.meta['src']=n; pool.append(c)
    objs=[]
    for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
        o=CLU.consensus(cl,len(files),'vote',0.5); v=o.meta['n']; o.score=v/len(files); best=0.0
        for c in single:
            if c.score>best and c.score>=0.05 and CLU.ciou(o,c)>=0.5: best=c.score
        o.meta={'v':v,'e40':best}; objs.append(o)
    return ph,objs
if __name__=='__main__':
    with Pool(4) as p: T=dict(p.map(photo,photos,chunksize=2))
    pickle.dump(T,open(Z+'/climb_val_table.pkl','wb'),protocol=4)
    allo=[o for ph in photos for o in T[ph] if o.area>=100]; med={v:float(np.median([o.meta['e40'] for o in allo if o.meta['v']==v])) for v in range(1,9)}
    print('objects by votes:',{v:sum(o.meta['v']==v for o in allo) for v in range(1,9)},' medians:',{v:round(m,3) for v,m in med.items()})
    def run(keep):
        res={}
        for ph in photos:
            sel=[o for o in T[ph] if keep(o)]; cv,ar,k=paint_labels(sel,-1,min_area=100); res[ph]=fast2.score_photo(cv,ar,G.g[ph])[0]
        return res
    base=run(lambda o:o.meta['v']>=4)
    rng=np.random.default_rng(3); S=[rng.choice(len(photos),size=90,replace=True) for _ in range(1000)]
    def pq(res,s=None): return pq_from([c for ph in (photos if s is None else [photos[i] for i in s]) for c in res[ph]])['pq']
    b0=pq(base); bb=np.array([pq(base,s) for s in S]); print('base (at least 4 of 8): %.4f'%b0)
    V={}
    for v in (1,2,3):
        V['add v=%d high-confidence half'%v]=lambda o,v=v:o.meta['v']>=4 or (o.meta['v']==v and o.meta['e40']>=med[v])
        V['add v=%d low-confidence half'%v]=lambda o,v=v:o.meta['v']>=4 or (o.meta['v']==v and o.meta['e40']<med[v])
    for v in (4,5,6,7):
        V['drop v=%d high-confidence half'%v]=lambda o,v=v:o.meta['v']>=4 and not (o.meta['v']==v and o.meta['e40']>=med[v])
        V['drop v=%d low-confidence half'%v]=lambda o,v=v:o.meta['v']>=4 and not (o.meta['v']==v and o.meta['e40']<med[v])
    V['add v<=3 with confidence >= 0.90']=lambda o:o.meta['v']>=4 or o.meta['e40']>=0.90
    V['add v<=3 with confidence >= 0.85']=lambda o:o.meta['v']>=4 or o.meta['e40']>=0.85
    V['drop v=4..6 with confidence < 0.80']=lambda o:o.meta['v']>=4 and not (o.meta['v']<=6 and o.meta['e40']<0.80)
    V['drop v=4..6 with confidence < 0.85']=lambda o:o.meta['v']>=4 and not (o.meta['v']<=6 and o.meta['e40']<0.85)
    print('%-40s objects  effect on PQ   90-photo sd   share of samples positive'%'change to the base')
    for n,k in V.items():
        r=run(k); d=np.array([pq(r,s) for s in S])-bb; nn=abs(sum(1 for o in allo if k(o))-sum(1 for o in allo if o.meta['v']>=4))
        print('%-40s %4d     %+.4f        %.4f        %.2f'%(n,nn,pq(r)-b0,d.std(),(d>0).mean()),flush=True)

"""Keep the object selection of the file vote, change only where each mask comes from.
Validation emulation of the eight-file vote (>= 4 of 8); masks: file-majority (baseline), three-model consensus,
e40 eight-view vote, e40 single view; optionally thickened. By annotator population."""
import sys,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
import cluster as CLU, fast2
from lib import load_train, pq_from
from subthick import thick
from ft_eval import SEL, by_pop
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
def swap(kept,source,minscore=0.3):
    out=[]; n=0
    src=[s for s in source if s.score>=minscore]
    for o in kept:
        best=None;bi=0.5
        for s in src:
            i=CLU.ciou(o,s)
            if i>=bi: bi=i;best=s
        if best is None: out.append(o)
        else:
            c=object.__new__(Cand); c.x0,c.y0,c.mask,c.area,c.score,c.meta=best.x0,best.y0,best.mask,best.area,o.score,{}; out.append(c); n+=1
    return out,n
def photo(ph):
    single=[c for c in M['e40'][ph] if c.meta.get('view',0)==0]
    d4e=d4_objects(M['e40'][ph]); three=multi_objects({m:M[m][ph] for m in ('e40','swins','swinb')},'mean8')
    files={'deployed':paint_file(single,0.85),'e40_d4_mean':paint_file(sc(d4e,'mean8'),0.85),'three_mean':paint_file(three,0.88),
           'swins_d4':paint_file(sc(d4_objects(M['swins'][ph]),'mean8'),0.88),'swinb_d4':paint_file(sc(d4_objects(M['swinb'][ph]),'mean8'),0.88),
           'ctrl31_d4':paint_file(sc(d4_objects(M['ctrl31'][ph]),'mean8'),0.85),'ctrl7':paint_file([c for c in M['ctrl7'][ph] if c.meta.get('view',0)==0],0.88),
           'head':paint_file([c for c in M['head'][ph] if c.meta.get('view',0)==0],0.90)}
    pool=[]
    for n,L in files.items():
        for c in L: c.meta['src']=n; pool.append(c)
    objs=[]
    for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
        o=CLU.consensus(cl,len(files),'vote',0.5); o.score=o.meta['n']/len(files); objs.append(o)
    kept=paint_labels(objs,4/len(files)-1e-9)[2]
    sc(d4e,'mean8')
    variants={'vote masks (baseline)':kept}
    variants['three-model consensus masks'],n1=swap(kept,three,0.3)
    variants['e40 eight-view masks'],n2=swap(kept,d4e,0.3)
    variants['e40 single-view masks'],n3=swap(kept,single,0.3)
    for k in list(variants): variants[k+' +thick .40']=[thick(o,0.40) for o in variants[k]]
    # deployed selection with other masks
    dep=paint_labels(single,0.85)[2]
    variants['DEPLOYED selection, own masks']=dep
    variants['DEPLOYED selection, three-model masks']=swap(dep,three,0.3)[0]
    variants['DEPLOYED selection, vote-file masks']=swap(dep,objs,0.0)[0]
    res={}
    for k,L in variants.items():
        cv,ar,kk=paint_labels(L,-1,min_area=100,order=L); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[k]=(e,len(ar))
    return ph,res,(len(kept),n1,n2,n3)
if __name__=='__main__':
    with Pool(4) as p: R=p.map(photo,photos,chunksize=2)
    pickle.dump(R,open(Z+'/maskswap.pkl','wb'))
    T=np.sum([r[2] for r in R],axis=0); print('kept objects %d; swapped: three-model %d, eight-view %d, single %d'%tuple(T))
    print('%-46s masks | '%'variant'+'  '.join('%7s'%s for s in SEL))
    for k in R[0][1]:
        res={ph:r[k][0] for ph,r,_ in R}; n=sum(r[k][1] for ph,r,_ in R)
        print('%-46s %4d  | '%(k,n)+'  '.join('%.4f '%x for x in by_pop(res,photos)),flush=True)

"""Does a slightly thicker mask help the file-level vote on the triple-annotated (target-group) validation records?
Vote of eight rebuilt validation 'files' (as in blend_val.py); thicker masks by a lower pixel-vote fraction or by blur+threshold."""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
import cluster as CLU, fast2
from lib import load_train, pq_from
from subthick import thick
from multiprocessing import Pool
Z='/home/user/work/an/zoo'; TARGET={'0101','0102','0104','0501'}
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb')),'ctrl31':load('ctrl31','val'),'ctrl7':load('ctrl7','val'),'head':load('head','val')}
photos=sorted(M['e40']); G=GT(photos); d,byimg,PH=load_train()
def paint_file(objs_by_photo,floor):
    out={}
    for ph in photos:
        cv,ar,kept=paint_labels(objs_by_photo[ph],floor); L=[]
        for i in range(len(ar)):
            m=(cv==i+1); ys,xs=np.nonzero(m); c=object.__new__(Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
            c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,m[y0:y1,x0:x1].copy(),int(ar[i]),1.0,{}; L.append(c)
        out[ph]=L
    return out
def single(name,floor): return paint_file({ph:[c for c in M[name][ph] if c.meta.get('view',0)==0] for ph in photos},floor)
def d4(name,agg,floor):
    O={ph:d4_objects(M[name][ph]) for ph in photos}
    for ph in photos:
        for o in O[ph]: o.score=o.meta[agg]
    return paint_file(O,floor)
def multi(names,agg,floor): return paint_file({ph:multi_objects({m:M[m][ph] for m in names},agg) for ph in photos},floor)
files={'deployed':single('e40',0.85),'e40_d4_mean':d4('e40','mean8',0.85),'three_mean':multi(('e40','swins','swinb'),'mean8',0.88),'swins_d4':d4('swins','mean8',0.88),
       'swinb_d4':d4('swinb','mean8',0.88),'ctrl31_d4':d4('ctrl31','mean8',0.85),'ctrl7':single('ctrl7',0.88),'head':single('head',0.90)}
names=list(files)
def blend(args):
    k,vote,blur=args; res={}; rows=0; a0=0
    for ph in photos:
        pool=[]
        for n in names:
            for c in files[n][ph]:
                c.meta['src']=n; pool.append(c)
        objs=[]
        for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(names),'vote',vote); o.score=o.meta['n']/len(names); objs.append(o)
        kept=paint_labels(objs,k/len(names)-1e-9)[2]
        if blur!=0.5: kept=[thick(o,blur) for o in kept]
        cv,ar,kk=paint_labels(kept,-1,min_area=100,order=kept); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[ph]=e; rows+=len(ar); a0+=int(ar.sum())
    return args,(res,rows,a0)
if __name__=='__main__':
    jobs=[(k,v,b) for k in (4,5) for (v,b) in ((0.5,0.5),(0.4,0.5),(0.34,0.5),(0.25,0.5),(0.6,0.5),(0.5,0.4),(0.5,0.35),(0.5,0.3),(0.4,0.4))]
    with Pool(4) as p: R=dict(p.map(blend,jobs,chunksize=1))
    pickle.dump(R,open(Z+'/blend_thick_val.pkl','wb'))
    sels={'all':lambda ph,e:True,'target':lambda ph,e:e[:4] in TARGET,'k=3':lambda ph,e:len(PH[ph])==3,'other':lambda ph,e:e[:4] not in TARGET,'0104':lambda ph,e:e[:4]=='0104','0501':lambda ph,e:e[:4]=='0501'}
    base_area={k:R[(k,0.5,0.5)][2] for k in (4,5)}
    print('agree pixvote blur | area x | '+'  '.join('%7s'%s for s in sels))
    for j in jobs:
        res,rows,a0=R[j]
        print('>=%d    %.2f   %.2f | %.3f  | '%(j[0],j[1],j[2],a0/base_area[j[0]])+'  '.join('%.4f '%pq_from([c for ph in photos for e,c in zip(PH[ph],res[ph]) if f(ph,e)])['pq'] for f in sels.values()),flush=True)

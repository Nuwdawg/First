"""File-level majority vote on validation: how many files should agree, and do more members help?"""
import sys,pickle,itertools; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
import cluster as CLU, fast2
Z='/home/user/work/an/zoo'
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb')),'ctrl31':load('ctrl31','val'),'ctrl7':load('ctrl7','val'),'focal':load('focal','val'),'head':load('head','val'),'paste':load('paste','val')}
photos=sorted(M['e40']); G=GT(photos); base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def paint_file(objs_by_photo,floor):
    """a 'file' = list of final masks per photo (as Cand objects with score 1)"""
    out={}
    for ph in photos:
        cv,ar,kept=paint_labels(objs_by_photo[ph],floor); L=[]
        for i in range(len(ar)):
            m=(cv==i+1); ys,xs=np.nonzero(m); c=object.__new__(Cand); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
            c.x0,c.y0,c.mask,c.area,c.score,c.meta=x0,y0,m[y0:y1,x0:x1].copy(),int(ar[i]),1.0,{}; L.append(c)
        out[ph]=L
    return out
files={}
def single(name,floor): return paint_file({ph:[c for c in M[name][ph] if c.meta.get('view',0)==0] for ph in photos},floor)
def d4(name,agg,floor):
    O={ph:d4_objects(M[name][ph]) for ph in photos}
    for ph in photos:
        for o in O[ph]: o.score=o.meta[agg]
    return paint_file(O,floor)
def multi(names,agg,floor): return paint_file({ph:multi_objects({m:M[m][ph] for m in names},agg) for ph in photos},floor)
files['deployed']=single('e40',0.85); files['e40_d4_mean']=d4('e40','mean8',0.85); files['e40_d4_maxf']=d4('e40','maxfrac',0.92)
files['swins_d4']=d4('swins','mean8',0.88); files['swinb_d4']=d4('swinb','mean8',0.88); files['ctrl31_d4']=d4('ctrl31','mean8',0.85)
files['three_mean']=multi(('e40','swins','swinb'),'mean8',0.88); files['three_maxf']=multi(('e40','swins','swinb'),'maxfrac',0.92)
files['ctrl7']=single('ctrl7',0.88); files['focal']=single('focal',0.90); files['head']=single('head',0.90); files['paste']=single('paste',0.88); files['swins_1']=single('swins',0.85); files['swinb_1']=single('swinb',0.88)
def score_file(f):
    res={}
    for ph in photos:
        cv,ar,k=paint_labels(f[ph],0.5); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[ph]=e
    return res
RES={n:score_file(f) for n,f in files.items()}
print('members:',{n:round(pq_of(r)['pq'],4) for n,r in RES.items()},flush=True)
def blend(names,thr,vote=0.5):
    res={};rows=0
    for ph in photos:
        pool=[]
        for n in names:
            for c in files[n][ph]:
                c.meta['src']=n; pool.append(c)
        objs=[]
        for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(names),'vote',vote); o.score=o.meta['n']/len(names); objs.append(o)
        cv,ar,k=paint_labels(objs,thr-1e-9); e,p=fast2.score_photo(cv,ar,G.g[ph]); res[ph]=e; rows+=len(ar)
    return res,rows
sets={'8 mixed (deployed, e40_d4_mean, three_mean, swins_d4, swinb_d4, ctrl31_d4, ctrl7, head)':['deployed','e40_d4_mean','three_mean','swins_d4','swinb_d4','ctrl31_d4','ctrl7','head'],
      '5 strongest':['three_mean','three_maxf','e40_d4_maxf','e40_d4_mean','swins_d4'],
      '6 single-model files':['deployed','swins_1','swinb_1','ctrl7','head','focal'],
      'all 14':list(files)}
for label,names in sets.items():
    line=[]
    for k in range(max(2,len(names)//4),len(names)*3//4+2):
        r,rows=blend(names,k/len(names)); d_,ci,pr=boot(r,base,photos,n=200); line.append('>=%d: %.4f (%d)'%(k,pq_of(r)['pq'],rows))
    print('%s | best member %.4f\n    %s'%(label,max(pq_of(RES[n])['pq'] for n in names),'  '.join(line)),flush=True)

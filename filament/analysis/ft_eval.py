"""Score the exp202 fine-tune outputs by annotator population (target = groups 0101/0102/0104/0501).
usage: ft_eval.py <output dir of the kernel>"""
import sys,json,pickle,glob,os,numpy as np; sys.path.insert(0,'/home/user/work/an')
from decode import *
from fast import Cand
from fast2 import GT, score_photo, paint_labels
from lib import load_train, pq_from
from multiprocessing import Pool
Z='/home/user/work/an/zoo'; TARGET={'0101','0102','0104','0501'}
d,byimg,PH=load_train()
SEL={'all':lambda ph,e:True,'target':lambda ph,e:e[:4] in TARGET,'k=3':lambda ph,e:len(PH[ph])==3,'other':lambda ph,e:e[:4] not in TARGET,'0104':lambda ph,e:e[:4]=='0104','0501':lambda ph,e:e[:4]=='0501'}
def by_pop(res,photos): return [pq_from([c for ph in photos for e,c in zip(PH[ph],res[ph]) if f(ph,e)])['pq'] for f in SEL.values()]
def load_simple(path):
    d=json.load(open(path)); return {fn.split('.')[0]:[Cand(c['rle'],c['score'],{'view':0,'src':'ft:0'}) for c in v] for fn,v in d.items()}
def load_zoo(path,name='ft'):
    d=json.load(open(path))['photographs']; out={}
    for fn,v in d.items():
        L=[]
        for mode in ('full','d4'):
            for c in v.get(mode,[]):
                view=int(c.get('view',0)) if mode=='d4' else 0
                L.append(Cand(c['rle'],c['score'],{'view':view,'model':name,'src':f'{name}:{view}'}))
        out[fn.split('.')[0]]=L
    return out
FLOORS=(0.75,0.8,0.85,0.88,0.9,0.92)
def single_job(path):
    V=load_simple(path); photos=sorted(V); G=GT(photos); rows=[]
    for f in FLOORS:
        res={}; n=0
        for ph in photos:
            cv,ar,k=paint_labels(V[ph],f); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e; n+=len(ar)
        rows.append((f,n/len(photos),by_pop(res,photos)))
    return path,rows
if __name__=='__main__':
    root=sys.argv[1]
    print('reference, deployed e40 single view @0.85:  all 0.4705  target 0.4570  k=3 0.4539  other 0.4842  0104 0.4317  0501 0.4815')
    caches=sorted(glob.glob(root+'/ft*/e[0-9][0-9][0-9]_cache.json'))
    with Pool(4) as p: R=p.map(single_job,caches,chunksize=1)
    print('%-28s floor masks/ph | '%'single view'+'  '.join('%7s'%s for s in SEL))
    for path,rows in R:
        for f,n,r in rows: print('%-28s %.2f  %5.2f   | '%('/'.join(path.split('/')[-2:]),f,n)+'  '.join('%.4f '%x for x in r))
        print()
    zp=root+'/d4_best/val_cache.json'
    if os.path.exists(zp):
        print('eight-view candidates of',open(root+'/d4_best/checkpoint.txt').read() if os.path.exists(root+'/d4_best/checkpoint.txt') else '?')
        V=load_zoo(zp); photos=sorted(V); G=GT(photos)
        M={'ft':V,'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
        def run(objs_by_photo,floor):
            res={}; n=0
            for ph in photos:
                cv,ar,k=paint_labels(objs_by_photo[ph],floor); e,p=score_photo(cv,ar,G.g[ph]); res[ph]=e; n+=len(ar)
            return n/len(photos),by_pop(res,photos)
        D4={ph:d4_objects(V[ph]) for ph in photos}
        for agg,fl in (('mean8',(0.8,0.85,0.88,0.9)),('maxfrac',(0.88,0.9,0.92,0.94))):
            for f in fl:
                for ph in photos:
                    for o in D4[ph]: o.score=o.meta[agg]
                n,r=run(D4,f); print('%-28s %.2f  %5.2f   | '%('ft d4 '+agg,f,n)+'  '.join('%.4f '%x for x in r),flush=True)
        for agg,fl in (('mean8',(0.85,0.88,0.9)),('maxfrac',(0.9,0.92,0.94))):
            O={ph:multi_objects({m:M[m][ph] for m in M},agg) for ph in photos}
            for f in fl:
                n,r=run(O,f); print('%-28s %.2f  %5.2f   | '%('ft+swins+swinb '+agg,f,n)+'  '.join('%.4f '%x for x in r),flush=True)

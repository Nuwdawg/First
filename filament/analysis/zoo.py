"""Model zoo: load candidate caches (val/test) for every Mask2Former-family model into pickles."""
import sys,json,pickle,os,time; sys.path.insert(0,'/home/user/work/an')
from fast import *
CA='/home/user/work/caches'
ZOO={ # name: (val cache, test cache)
 'e40':      (CA+'/exp147/val_cache.json', CA+'/e040test/e040_test_cache.json'),
 'base20':   (CA+'/sol-exp184a-vote-caches-t4/base_e020_val_cache.json', CA+'/sol-exp184a-vote-caches-t4/base_e020_test_cache.json'),
 'ctrl31':   (CA+'/sol-exp184a-vote-caches-t4/control_e031_val_cache.json', CA+'/sol-exp184a-vote-caches-t4/control_e031_test_cache.json'),
 'qual31':   (CA+'/sol-exp184a-vote-caches-t4/quality_e031_val_cache.json', CA+'/sol-exp184a-vote-caches-t4/quality_e031_test_cache.json'),
 'ctrl7':    (CA+'/codex-extra-model-caches/control7/control7_val_cache.json', CA+'/codex-extra-model-caches/control7/control7_test_cache.json'),
 'focal':    (CA+'/codex-extra-model-caches/focal/focal_val_cache.json', CA+'/codex-extra-model-caches/focal/focal_test_cache.json'),
 'head':     (CA+'/codex-quality-model-caches/head/head_val_cache.json', CA+'/codex-quality-model-caches/head/head_test_cache.json'),
 'paste':    (CA+'/codex-quality-model-caches/paste/paste_val_cache.json', CA+'/codex-quality-model-caches/paste/paste_test_cache.json'),
}
PK='/home/user/work/an/zoo'
def build(name,split):
    out=f'{PK}/{name}_{split}.pkl'
    if os.path.exists(out): return out
    path=ZOO[name][0 if split=='val' else 1]
    d=json.load(open(path))['photographs']; res={}
    for fn,v in d.items():
        L=[]
        for mode in ('full','d4'):
            for c in v.get(mode,[]):
                view=int(c.get('view',0)) if mode=='d4' else 0
                o=Cand(c['rle'],c['score'],{'p':c.get('p'),'m':c.get('m'),'view':view,'model':name,'src':f'{name}:{view}'})
                L.append(o)
        res[fn.split('.')[0]]=L
    pickle.dump(res,open(out,'wb'),protocol=4); return out
def load(name,split): return pickle.load(open(build(name,split),'rb'))
if __name__=='__main__':
    os.makedirs(PK,exist_ok=True)
    from multiprocessing import Pool
    jobs=[(n,s) for s in ('val','test') for n in ZOO]
    t0=time.time()
    with Pool(4) as p:
        for r in p.starmap(build,jobs): print(r,round(time.time()-t0),flush=True)

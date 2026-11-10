import sys,pickle,json,os,time; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast import Cand
from fast2 import GT, score_photo
from lib import pq_from
from build_sub import check_sub
Z='/home/user/work/an/zoo'
def cache_to_pkl(path,out):
    if os.path.exists(out): return pickle.load(open(out,'rb'))
    d=json.load(open(path))['photographs']; V={}
    for fn,v in d.items():
        V[fn.split('.')[0]]=[Cand(c['rle'],c['score'],{'src':'full' if mode=='full' else 'd4_%s'%c['view'],'view':int(c.get('view',0)) if mode=='d4' else 0}) for mode in ('full','d4') for c in v.get(mode,[])]
    pickle.dump(V,open(out,'wb'),protocol=4); return V
VAL={'ctrl31':lambda:load('ctrl31','val'),'e40':lambda:load('e40','val'),'swins':lambda:pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':lambda:pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
TEST={'ctrl31':lambda:load('ctrl31','test'),'e40':lambda:load('e40','test'),
      'swins':lambda:cache_to_pkl('/home/user/work/caches/exp201s2/cache_swins/model_e050_test_cache.json',Z+'/swinsALL_test.pkl'),
      'swinb':lambda:cache_to_pkl('/home/user/work/caches/codex-swinb-paired-views/swinb_full_test/swinb_full_test_test_cache.json',Z+'/swinbALL_test.pkl')}
models=sys.argv[1].split(','); agg=sys.argv[2]; floor=float(sys.argv[3]); out=sys.argv[4]
V={m:VAL[m]() for m in models}; photos=sorted(V[models[0]]); G=GT(photos); ent=[]; nv=0
for ph in photos:
    cv,ar,k=rule_multi({m:V[m][ph] for m in models},agg,floor); e,p=score_photo(cv,ar,G.g[ph]); ent+=e; nv+=len(ar)
a=pq_from(ent); print('VAL replay (hold-out twins) %s %s floor %.2f: PQ %.4f (%d/%d/%d), %.2f masks per photo'%('+'.join(models),agg,floor,a['pq'],a['tp'],a['fp'],a['fn'],nv/len(photos)),flush=True)
del V
T={m:TEST[m]() for m in models}; cvs={}; nt=0
for ph in sorted(T[models[0]]):
    cv,ar,k=rule_multi({m:T[m].get(ph,[]) for m in models},agg,floor); cvs[ph]=(cv,len(ar)); nt+=len(ar)
print('TEST rows',write_submission(cvs,out),'%.2f masks per photo'%(nt/len(cvs))); print(check_sub(out))

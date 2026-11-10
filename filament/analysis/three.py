import sys,time,pickle,json,itertools; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects
# swins-long val into pickle
import os
if not os.path.exists('/home/user/work/an/zoo/swinsL_val.pkl'):
    d=json.load(open('/home/user/work/caches/codex-swins-heldout-views/swins/swins_val_cache.json'))['photographs']; V={}
    for fn,v in d.items():
        V[fn.split('.')[0]]=[Cand(c['rle'],c['score'],{'src':'full' if mode=='full' else 'd4_%s'%c['view'],'view':int(c.get('view',0)) if mode=='d4' else 0}) for mode in ('full','d4') for c in v.get(mode,[])]
    pickle.dump(V,open('/home/user/work/an/zoo/swinsL_val.pkl','wb'),protocol=4)
M={'e40':load('e40','val'),'swinsL':pickle.load(open('/home/user/work/an/zoo/swinsL_val.pkl','rb')),'swinbL':pickle.load(open('/home/user/work/an/zoo/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos); _,_,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
PM={n:{ph:d4_objects(M[n][ph]) for ph in photos} for n in M}
for n in PM:
    for ph in photos:
        for o in PM[n][ph]: o.meta['src']=n
def report(label,O,FL):
    rows=[]
    for f in FL:
        r=run(O,G,photos,f); rows.append((pq_of(r)['pq'],f,r,pq_from([c for p in photos if kk[p]>=2 for c in r[p]])['pq']))
    b=max(rows,key=lambda t:t[0]); d_,ci,pr=boot(b[2],base,photos,n=500)
    print('%-34s best %.4f @%.2f %+.4f [%+.4f,%+.4f] P %.2f | %s | k>=2 best %.4f'%(label,b[0],b[1],d_,ci[0],ci[1],pr,' '.join('%.2f:%.4f'%(f,p) for p,f,_,_ in rows),max(r[3] for r in rows)),flush=True)
for combo in [('e40','swinsL'),('e40','swinbL'),('swinsL','swinbL'),('e40','swinsL','swinbL')]:
    for key in ('mean8','maxfrac'):
        for n in combo:
            for ph in photos:
                for o in PM[n][ph]: o.score=o.meta[key]
        CL2={ph:cluster([o for n in combo for o in PM[n][ph]],link=0.5,min_score=0.01) for ph in photos}
        O={ph:[consensus(k,len(combo),'vote',0.5) for k in cl] for ph,cl in CL2.items()}
        for ph in photos:
            for o in O[ph]: o.score=o.meta['sum']/len(combo)
        report('+'.join(combo)+' '+key+' mean',O,(0.7,0.75,0.8,0.85,0.88,0.9,0.92))

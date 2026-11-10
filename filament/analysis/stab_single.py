"""Single-view stability scores from cached evidence: decoder-layer agreement and mask-threshold stability."""
import sys,time,pickle,json; sys.path.insert(0,'/home/user/work/an')
from zoo_eval import *
from fast import rle_fix
vc=json.load(open('/home/user/work/caches/exp147/val_cache.json'))['photographs']
photos=sorted(k.split('.')[0] for k in vc); G=GT(photos)
d,byimg,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
C={}
for fn,v in vc.items():
    L=[]
    for c in v['full']:
        o=Cand(c['rle'],c['score'],{})
        lo=mu.area(rle_fix(c['rle_low'])); hi=mu.area(rle_fix(c['rle_high']))
        o.meta=dict(p=c['p'],m=c['m'],layers=np.array(c['layers']),sharp=float(hi)/max(float(lo),1))
        L.append(o)
    C[fn.split('.')[0]]=L
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def report(label,fn,FL):
    for ph in photos:
        for o in C[ph]: o.score=fn(o.meta)
    rows=[]
    for f in FL:
        r=run(C,G,photos,f); a=pq_of(r); m=pq_from([c for ph in photos if kk[ph]>=2 for c in r[ph]]); rows.append((a['pq'],f,r,a,m['pq']))
    best=max(rows,key=lambda t:t[0]); d_,ci,pr=boot(best[2],base,photos,n=500)
    print('%-30s best %.4f @%.2f (%d/%d/%d) %+.4f [%+.4f,%+.4f] | %s | k>=2 best %.4f'%(label,best[0],best[1],best[3]['tp'],best[3]['fp'],best[3]['fn'],d_,ci[0],ci[1],' '.join('%.2f:%.4f'%(f,p) for p,f,_,_,_ in rows),max(r[4] for r in rows)),flush=True)
FL=(0.6,0.7,0.75,0.8,0.85,0.88,0.9,0.92)
report('p*m (deployed)',lambda m:m['p']*m['m'],FL)
report('p',lambda m:m['p'],(0.8,0.85,0.88,0.9,0.92,0.94,0.96))
report('mean(last3 layers)*m',lambda m:m['layers'][-3:].mean()*m['m'],FL)
report('min(last3 layers)*m',lambda m:m['layers'][-3:].min()*m['m'],FL)
report('mean(last5 layers)*m',lambda m:m['layers'][-5:].mean()*m['m'],FL)
report('min(last5 layers)*m',lambda m:m['layers'][-5:].min()*m['m'],FL)
report('p*m*sharp^0.5',lambda m:m['p']*m['m']*m['sharp']**0.5,(0.5,0.55,0.6,0.65,0.7,0.75,0.8))
report('p*sharp',lambda m:m['p']*m['sharp'],(0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75))
sh=np.array([o.meta['sharp'] for ph in photos for o in C[ph] if o.meta['p']>0.5]); print('sharpness (area hi/area lo) quantiles for p>0.5:',np.quantile(sh,[.05,.25,.5,.75,.95]).round(3))

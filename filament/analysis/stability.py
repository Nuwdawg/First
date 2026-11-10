"""Stability score: e40 confidence x fraction of all sources (any model, any view) that detect the object."""
import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
names=sys.argv[1].split(',')
V={n:load(n,'val') for n in names}; photos=sorted(V['e40']); G=GT(photos)
d,byimg,PH=load_train(); kk={ph:len(PH[ph]) for ph in photos}
srcs=sorted({c.meta['src'] for n in names for c in V[n][photos[0]]}|{c.meta['src'] for n in names for c in V[n][photos[1]]})
nsrc=sum(8 if any(c.meta['view']>0 for c in V[n][photos[0]]+V[n][photos[1]]+V[n][photos[2]]) else 1 for n in names)
print('models',names,'n sources',nsrc,flush=True)
t0=time.time()
def clus(ph):
    # seed order: e40 candidates first (by score), then others, so clusters are anchored on e40 masks
    L=[c for n in names for c in V[n][ph]]
    for c in L: c.meta['_key']=(c.meta['model']=='e40', c.score)
    cl=[]
    for c in sorted(L,key=lambda c:(-int(c.meta['model']=='e40'),-c.score)):
        if c.area<20: continue
        best=None;bi=0.5
        for k in cl:
            if any(m.meta['src']==c.meta['src'] for m in k.members): continue
            v=ciou(k.seed,c)
            if v>=bi: bi=v;best=k
        if best is None: cl.append(Cluster(c))
        else: best.members.append(c)
    return cl
CL={ph:clus(ph) for ph in photos}; print('clustered',round(time.time()-t0),flush=True)
def build(maskfrom,scorefn):
    O={}
    for ph in photos:
        objs=[]
        for k in CL[ph]:
            e=[m for m in k.members if m.meta['model']=='e40']
            if not e: continue                       # object must be seen by e40 (calibrated score scale)
            sub=Cluster(e[0]); sub.members=e if maskfrom=='e40' else k.members
            o=consensus(sub,8,'vote',0.6)
            o.meta.update(e_n=len(e),e_max=max(m.score for m in e),e_mean8=sum(m.score for m in e)/8,all_n=len(k.members))
            o.score=scorefn(o.meta); objs.append(o)
        O[ph]=objs
    return O
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def report(label,O,FL):
    rows=[]
    for f in FL:
        r=run(O,G,photos,f); a=pq_of(r); m=pq_from([c for ph in photos if kk[ph]>=2 for c in r[ph]]); rows.append((a['pq'],f,r,a,m['pq']))
    best=max(rows,key=lambda t:t[0]); d,ci,pr=boot(best[2],base,photos,n=600)
    print('%-34s best %.4f @%.2f (%d/%d/%d) %+.4f [%+.4f,%+.4f] P %.2f | all: %s | k>=2: %s'%(label,best[0],best[1],best[3]['tp'],best[3]['fp'],best[3]['fn'],d,ci[0],ci[1],pr,' '.join('%.2f:%.4f'%(f,p) for p,f,_,_,_ in rows),' '.join('%.4f'%m for _,_,_,_,m in rows)),flush=True)
FL=(0.8,0.85,0.88,0.9,0.92,0.94)
report('e40 max*frac8 (reference)',build('e40',lambda m:m['e_max']*m['e_n']/8),FL)
report('e40 max * frac_all',build('e40',lambda m:m['e_max']*m['all_n']/nsrc),(0.7,0.75,0.8,0.85,0.88,0.9,0.92))
report('e40 max*frac8 * frac_all^0.5',build('e40',lambda m:m['e_max']*m['e_n']/8*(m['all_n']/nsrc)**0.5),(0.7,0.75,0.8,0.85,0.88,0.9,0.92))
report('e40 mean8 * frac_all',build('e40',lambda m:m['e_mean8']*m['all_n']/nsrc),(0.6,0.65,0.7,0.75,0.8,0.85,0.88))
report('allmask: e40 max * frac_all',build('all',lambda m:m['e_max']*m['all_n']/nsrc),(0.7,0.75,0.8,0.85,0.88,0.9,0.92))
print('time',round(time.time()-t0))

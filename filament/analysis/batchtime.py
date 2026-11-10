import sys; sys.path.insert(0,'/home/user/work/an')
from lib import *
import datetime as dt
d,byimg,photos=load_train()
def t(ph): return dt.datetime.strptime(ph[:14],'%Y%m%d%H%M%S').timestamp()/86400
P=sorted(photos,key=t); T=np.array([t(p) for p in P])
gb=[frozenset(e[:4] for e in photos[p]) for p in P]      # group+batch set
aset=[frozenset(e.split('-')[0] for e in photos[p]) for p in P]
k=[len(photos[p]) for p in P]
import random
same_gb=[];same_k=[];gaps=[]
for i in range(len(P)):
    j=np.argsort(np.abs(T-T[i]))[1]
    gaps.append(abs(T[j]-T[i])); same_gb.append(gb[i]==gb[j]); same_k.append(k[i]==k[j])
gaps=np.array(gaps);same_gb=np.array(same_gb);same_k=np.array(same_k)
rng=np.random.default_rng(0); perm=rng.permutation(len(P))
print('chance same group-batch',np.mean([gb[i]==gb[j] for i,j in zip(range(len(P)),perm)]).round(3),'chance same k',np.mean([k[i]==k[j] for i,j in zip(range(len(P)),perm)]).round(3))
for lo,hi in [(0,0.5),(0.5,1.5),(1.5,3),(3,7),(7,1000)]:
    s=(gaps>=lo)&(gaps<hi)
    print('nearest gap %g-%g d: n %d same group-batch %.3f same multiplicity %.3f'%(lo,hi,s.sum(),same_gb[s].mean(),same_k[s].mean()))
# sequence view: first 60 photos in time with their group-batch
for p,g,kk in list(zip(P,gb,k))[:45]: print(p,sorted(g),kk)

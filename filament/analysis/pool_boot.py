"""Each pool rule against the replay of the eight-file vote on the whole validation set: mean difference and its
scatter on a random 90-photograph sample (photo bootstrap), and the chance of beating the vote by 0.0025."""
import sys,pickle,numpy as np; sys.path.insert(0,'/home/user/work/an')
from lib import pq_from
Z='/home/user/work/an/zoo'
P=pickle.load(open(Z+'/pool13_val.pkl','rb')); B=pickle.load(open(Z+'/blend_thick_val.pkl','rb')); R3,K=pickle.load(open(Z+'/bysub3.pkl','rb'))
photos=sorted(P); E={'vote of 8 (>=4)':B[(4,0.5,0.5)][0],'vote of 8 (>=5)':B[(5,0.5,0.5)][0],'single @0.85':R3['single@0.85'],'three mean .88 (F)':R3['three_mean8@0.88'],'three maxfrac .92 (G)':R3['three_maxfrac@0.92'],'e40 d4 maxfrac .92 (C)':R3['e40d4_maxfrac@0.92']}
for k in P[photos[0]]: E[k]={ph:P[ph][k][0] for ph in photos}
rng=np.random.default_rng(1); S=[rng.choice(len(photos),size=90,replace=True) for _ in range(1500)]
def pq(name,s): return pq_from([c for i in s for c in E[name][photos[i]]])['pq']
base=np.array([pq('vote of 8 (>=4)',s) for s in S])
print('%-24s  val PQ   minus vote: mean    sd     P(> +0.0025)'%'rule')
rows=[]
for k in E:
    v=np.array([pq(k,s) for s in S]); d=v-base
    rows.append((k,pq_from([c for ph in photos for c in E[k][ph]])['pq'],d.mean(),d.std(),(d>0.0025).mean()))
for r in sorted(rows,key=lambda r:-r[4]): print('%-24s  %.4f   %+.4f   %.4f   %.2f'%r,flush=True)

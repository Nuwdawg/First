import sys,glob,os; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub
from lib import *
S='/home/user/work/subs/'
F={'best50':'blend41_thr50.csv','best60':'blend41_thr60.csv','deployed':'repro_e40_floor085.csv','F':'cand_F_three_mean8_088.csv','C':'cand_C_e40_d4_maxfrac_092.csv','S9k4':'voteS9_k4.csv','DM':'deployed_sel_three_masks.csv','b16_50':'blend16_thr50.csv'}
for p in sorted(glob.glob(S+'pool13/*.csv')): F[os.path.basename(p)[:-4]]='pool13/'+os.path.basename(p)
R={k:read_sub(S+v) for k,v in F.items()}
photos=sorted(set().union(*[set(r) for r in R.values()]))
import itertools,numpy as np
names=list(F); A=np.eye(len(names))
for i,j in itertools.combinations(range(len(names)),2):
    A[i,j]=A[j,i]=pq_from([pq_counts(iou_matrix(R[names[i]].get(ph,[]),R[names[j]].get(ph,[]))) for ph in photos])['pq']
import pickle; pickle.dump((names,A),open('/home/user/work/an/zoo/agree_pool.pkl','wb'))
print('%-16s masks | vs best50 best60 deployed F     C     S9k4  five85 V5d4_k3'%'file')
ref=[names.index(x) for x in ('best50','best60','deployed','F','C','S9k4','five85','V5d4_k3')]
for i,n in enumerate(names): print('%-16s %5d | '%(n,sum(len(v) for v in R[n].values()))+'  '.join('%.3f'%A[i,r] for r in ref))

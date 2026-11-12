"""How different are the candidate files from the current best and from each other? Agreement = PQ of one file scored
against the other as if it were ground truth (1.0 = identical); also the number of masks."""
import sys; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub
from lib import *
S='/home/user/work/subs/'
F={'best50':'blend41_thr50.csv','best60':'blend41_thr60.csv','deployed':'repro_e40_floor085.csv','F':'cand_F_three_mean8_088.csv','G':'cand_G_three_maxfrac_092.csv',
   'C':'cand_C_e40_d4_maxfrac_092.csv','A':'cand_A_e40_d4_mean8_085.csv','F85':'cand_F85_three_mean8_085.csv','S9k4':'voteS9_k4.csv','S9k5':'voteS9_k5.csv',
   'DM':'deployed_sel_three_masks.csv','b16_50':'blend16_thr50.csv','b16_44':'blend16_thr44.csv','D':'cand_D_swinsALL_mean8_088.csv','E':'cand_E_swinbALL_mean8_088.csv',
   'H':'cand_H_four_maxfrac_090.csv','swinb1':'m13/swinb.csv','ftA6':'m13/ftA6.csv','thr38':'blend8_thr38.csv'}
R={k:read_sub(S+v) for k,v in F.items()}
photos=sorted(set().union(*[set(r) for r in R.values()]))
def agree(a,b): return pq_from([pq_counts(iou_matrix(R[a].get(ph,[]),R[b].get(ph,[]))) for ph in photos])['pq']
names=list(F)
print('%-9s masks  vs best50  vs best60  vs deployed  vs S9k4  vs F'%'file')
for n in names:
    print('%-9s %5d   %.3f      %.3f      %.3f        %.3f    %.3f'%(n,sum(len(v) for v in R[n].values()),agree(n,'best50'),agree(n,'best60'),agree(n,'deployed'),agree(n,'S9k4'),agree(n,'F')),flush=True)

"""How different are the candidate test files from each other? Agreement = PQ of one file scored against another."""
import sys,glob,os; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub
from lib import *
S='/home/user/work/subs/'
files={'deployed':S+'repro_e40_floor085.csv','prev7src':glob.glob(S+'dl_58160222/*')[0],'A':S+'cand_A_e40_d4_mean8_085.csv','C':S+'cand_C_e40_d4_maxfrac_092.csv','D':S+'cand_D_swinsALL_mean8_088.csv','E':S+'cand_E_swinbALL_mean8_088.csv','F':S+'cand_F_three_mean8_088.csv','G':S+'cand_G_three_maxfrac_092.csv','G94':S+'cand_G94_three_maxfrac_094.csv','F85':S+'cand_F85_three_mean8_085.csv'}
subs={k:read_sub(v) for k,v in files.items()}
photos=sorted(set().union(*[set(s) for s in subs.values()]))
names=list(files); print('rows:',{k:sum(len(v) for v in subs[k].values()) for k in names})
def agree(a,b):
    cs=[]
    for ph in photos:
        cs.append(pq_counts(iou_matrix(subs[a].get(ph,[]),subs[b].get(ph,[]))))
    r=pq_from(cs); return r['pq'],r['rq']
print('agreement (PQ of column file scored against row file):')
print('%-9s'%''+' '.join('%-8s'%n for n in names))
for a in names:
    print('%-9s'%a+' '.join('%-8.3f'%agree(a,b)[0] for b in names),flush=True)

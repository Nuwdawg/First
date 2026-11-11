"""Does a stricter match threshold reproduce the leaderboard triple 0.39 / 0.41 / 0.40 (floors 0.70 / 0.85 / 0.90)?"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
C=load_val_cands('full'); photos=list(C); G=GT(photos)
IOU={}
for f in (0.7,0.85,0.9):
    L=[]
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,f)
        for e,items in G.g[ph]: L.append(iou_mat(cv,ar,items))
    IOU[f]=L
def pq_tau(L,tau,one=False):
    cs=[]
    for M in L:
        ng,npr=M.shape
        if ng==0: cs.append((0,0,npr,0)); continue
        if npr==0: cs.append((0,0,0,ng)); continue
        hit=M>tau; cs.append((float(M[hit].sum()),int(hit.sum()),int((hit.sum(0)==0).sum()),int((hit.sum(1)==0).sum())))
    return pq_from(cs)['pq']
print('tau    | floor 0.70  0.85  0.90   (leaderboard: 0.39 0.41 0.40)')
for tau in (0.5,0.52,0.54,0.55,0.56,0.58,0.6,0.65):
    print('%.2f   | %.4f %.4f %.4f'%(tau,pq_tau(IOU[0.7],tau),pq_tau(IOU[0.85],tau),pq_tau(IOU[0.9],tau)))

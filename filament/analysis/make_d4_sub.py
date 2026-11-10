import sys,pickle,time; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast2 import GT, score_photo
from lib import pq_from
from build_sub import check_sub
agg=sys.argv[1]; floor=float(sys.argv[2]); out=sys.argv[3]
# replay on validation first
V=load('e40','val'); photos=sorted(V); G=GT(photos); ent=[]
for ph in photos:
    cv,ar,k=rule_d4(V[ph],agg,floor); e,p=score_photo(cv,ar,G.g[ph]); ent+=e
a=pq_from(ent); print('VAL replay %s floor %.2f: PQ %.4f (%d/%d/%d)'%(agg,floor,a['pq'],a['tp'],a['fp'],a['fn']),flush=True)
T=load('e40','test'); cvs={}
for ph in sorted(T):
    cv,ar,k=rule_d4(T[ph],agg,floor); cvs[ph]=(cv,len(ar))
print('rows',write_submission(cvs,out)); print(check_sub(out))

"""Validation PQ by annotator group. Triple-annotated photographs occur only in groups 0101, 0102, 0104, 0501;
if the test set is triple-annotated, those are the annotators who score it ("target")."""
import sys,pickle,collections; sys.path.insert(0,'/home/user/work/an')
from lib import pq_from, load_train
R,K=pickle.load(open('/home/user/work/an/zoo/bysub3.pkl','rb'))
d,byimg,PH=load_train(); photos=sorted(K)
TARGET={'0101','0102','0104','0501'}
def agg(name,sel):
    ent=[c for ph in photos for e,c in zip(PH[ph],R[name][ph]) if sel(ph,e)]
    return pq_from(ent),len(ent)
grp=collections.Counter(e[:4] for ph in photos for e in PH[ph])
print('validation records by group:',dict(sorted(grp.items())))
names=['single@0.85','three_mean8@0.88','three_maxfrac@0.92',"single@0.85 (0, 0, ('dilate', 'cross', 1))","single@0.85 (0, 0, ('erode', 'cross', 1))","three_mean8@0.88 (0, 0, ('dilate', 'cross', 1))","three_mean8@0.88 (0, 0, ('erode', 'cross', 1))"]
print('%-10s %4s | single  3mean8  3maxfr | 1 dil    1 ero  | 3 dil    3 ero'%('group','rec'))
for g in sorted(grp):
    row=[agg(n,lambda ph,e:e[:4]==g) for n in names]
    print('%-10s %4d | %.4f  %.4f  %.4f | %.4f  %.4f | %.4f  %.4f'%(g,row[0][1],*[r[0]['pq'] for r in row]))
for lab,sel in (('target',lambda ph,e:e[:4] in TARGET),('other',lambda ph,e:e[:4] not in TARGET),('target k=3',lambda ph,e:e[:4] in TARGET and K[ph]==3),('target k<3',lambda ph,e:e[:4] in TARGET and K[ph]<3)):
    row=[agg(n,sel) for n in names]
    print('%-10s %4d | %.4f  %.4f  %.4f | %.4f  %.4f | %.4f  %.4f'%(lab,row[0][1],*[r[0]['pq'] for r in row]))
print('\nfloor sweep, target-group records vs other records')
for n in R:
    if '(' in n: continue
    a,na=agg(n,lambda ph,e:e[:4] in TARGET); b,nb=agg(n,lambda ph,e:e[:4] not in TARGET); c,nc=agg(n,lambda ph,e:K[ph]==3)
    print('%-24s target %.4f (%d/%d/%d)   other %.4f   k=3 %.4f'%(n,a['pq'],a['tp'],a['fp'],a['fn'],b['pq'],c['pq']))

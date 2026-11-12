import sys,os; sys.path.insert(0,'/home/user/work/an')
from blendw import *
pool={}; pool.update(load_tier('lb41',1.0)); pool.update(load_tier('lb40',0.6)); pool.update(load_tier('lb39',0.3))
kept=['58219271','58186186','58160222','58159470','58127006','57816701','56648050','56309562','58219268','58194117','58185837','58173955','58159386','58150014','57993307','56423657','56323263','56321802','56312952','56282198','56273757','56212792']   # distinct files found by blendw.py
photos=sorted(set().union(*[set(s) for s,_ in pool.values()]))
k8=[k for k in kept if pool[k][1]==1.0]
# (1) family-weighted: exp169 family (seven 0.41 files) weight 2, exp162 family (two) weight 1.5
fw=dict(pool); fw['58127006']=(pool['58127006'][0],2.0); fw['56648050']=(pool['56648050'][0],1.5)
build(fw,k8,photos,0.5,S+'blend8_famw_thr50.csv')
# (2) eight 0.41 files plus the YOLO-based files at full weight
yy=dict(pool)
for k in kept:
    if pool[k][1]==0.3: yy[k]=(pool[k][0],1.0)
ky=k8+[k for k in kept if pool[k][1]==0.3]
build(yy,ky,photos,0.5,S+'blend8_yolo_thr50.csv'); build(yy,ky,photos,0.4,S+'blend8_yolo_thr40.csv')
# (3) nineteen files of the 0.41 and 0.40 tiers at equal weight
eq=dict(pool); k19=[k for k in kept if pool[k][1]>=0.6]
for k in k19: eq[k]=(pool[k][0],1.0)
build(eq,k19,photos,0.5,S+'blend19_eq_thr50.csv'); build(eq,k19,photos,0.4,S+'blend19_eq_thr40.csv')
from build_sub import read_sub
a=read_sub(S+'blend41_thr50.csv')
for name in ['blend8_famw_thr50','blend8_yolo_thr50','blend8_yolo_thr40','blend19_eq_thr50','blend19_eq_thr40']:
    b=read_sub(S+name+'.csv'); print(name,'agreement with blend41_thr50: %.3f'%pq_from([pq_counts(iou_matrix(a.get(ph,[]),b.get(ph,[]))) for ph in set(a)|set(b)])['pq'])

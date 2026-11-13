"""Second-opinion attributes for the vote objects: how many of five independently built consensus files contain each
object, and which objects those files agree on that the eight-file vote does not have. Writes the next variants,
all relative to c1 (v>=5, or v==4 with deployed confidence >= the group median)."""
import sys,os,pickle,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
import climb
from build_sub import read_sub, check_sub
from fast import Cand
import cluster as CLU
S='/home/user/work/subs/'; CONS=['three85','five85','V5d4_k3','ftswinsswinb85','d4_swinb']
T=climb.table()
allo=[o for ph in T for o in T[ph] if o.area>=100]
med={v:float(np.median([o.meta['e40'] for o in allo if o.meta['v']==v])) for v in range(1,9)}
print('medians',{v:round(m,3) for v,m in med.items()})
F={n:read_sub(S+f'pool13/{n}.csv') for n in CONS}
extra={}
for ph in sorted(T):
    C={n:[Cand(r,1.0,{'src':n}) for r in F[n].get(ph,[])] for n in CONS}
    for o in T[ph]: o.meta['cons']=sum(any(CLU.ciou(o,c)>=0.5 for c in C[n]) for n in CONS)
    rest=[c for n in CONS for c in C[n] if not any(CLU.ciou(c,o)>=0.5 for o in T[ph])]
    new=[]
    for cl in CLU.cluster(rest,link=0.5,min_score=0.0):
        o=CLU.consensus(cl,len(CONS),'vote',0.5); k=o.meta['n']; o.score=0.0; o.meta={'v':0,'e40':0.0,'cons':k,'src':'cons'}; new.append(o)
    extra[ph]=new
pickle.dump((T,extra),open('/home/user/work/an/zoo/climb_table2.pkl','wb'),protocol=4)
c1=lambda o:o.meta['v']>=5 or (o.meta['v']==4 and o.meta['e40']>=med[4])
inc=[o for o in allo if c1(o)]; exc=[o for o in allo if not c1(o)]
print('c1 objects',len(inc),'by number of consensus files containing them:',dict(sorted(collections.Counter(o.meta['cons'] for o in inc).items())))
for v in (4,5,6,7): print('   v=%d in c1:'%v,dict(sorted(collections.Counter(o.meta['cons'] for o in inc if o.meta['v']==v).items())))
print('vote objects outside c1',len(exc),':',dict(sorted(collections.Counter(o.meta['cons'] for o in exc).items())))
for v in (1,2,3,4): print('   v=%d outside c1:'%v,dict(sorted(collections.Counter(o.meta['cons'] for o in exc if o.meta['v']==v).items())))
ne=[o for ph in extra for o in extra[ph] if o.area>=100]; print('objects in no 0.41 file:',len(ne),dict(sorted(collections.Counter(o.meta['cons'] for o in ne).items())))
def write(keep,name,addnew=None):
    import csv
    from fast2 import paint_labels, canvas_to_rles
    rows=0
    with open(S+f'climb/{name}.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(T):
            sel=[o for o in T[ph] if keep(o)]
            if addnew:
                for o in extra[ph]:
                    if addnew(o):
                        o.score=-1.0; sel.append(o)       # painted last
            cv,ar,k=paint_labels(sel,-2,min_area=100)
            for i,r in enumerate(canvas_to_rles(cv,len(ar))): w.writerow([f'{ph}_{i+1}',r['counts'].decode()]); rows+=1
    print(name,'rows',rows,check_sub(S+f'climb/{name}.csv'),flush=True)
V={'d1_c1_minus_v5': lambda o:o.meta['v']>=6 or (o.meta['v']==4 and o.meta['e40']>=med[4]),
   'd2_c1_plus_v3hi': lambda o:c1(o) or (o.meta['v']==3 and o.meta['e40']>=med[3]),
   'd3_c1_minus_v5_plus_v3hi': lambda o:o.meta['v']>=6 or (o.meta['v']==4 and o.meta['e40']>=med[4]) or (o.meta['v']==3 and o.meta['e40']>=med[3]),
   'd4_c1_minus_v5lo': lambda o:o.meta['v']>=6 or (o.meta['v']==5 and o.meta['e40']>=med[5]) or (o.meta['v']==4 and o.meta['e40']>=med[4]),
   'd5_c1_minus_v5hi': lambda o:o.meta['v']>=6 or (o.meta['v']==5 and o.meta['e40']<med[5]) or (o.meta['v']==4 and o.meta['e40']>=med[4]),
   'd6_c1_minus_cons0': lambda o:c1(o) and not (o.meta['cons']==0 and o.meta['v']<=7),
   'd7_c1_plus_cons5': lambda o:c1(o) or o.meta['cons']==5,
   'd8_c1_minus_cons0_plus_cons5': lambda o:(c1(o) and not (o.meta['cons']==0 and o.meta['v']<=7)) or o.meta['cons']==5}
for n,k in V.items(): write(k,n)
write(c1,'d9_c1_plus_new4',addnew=lambda o:o.meta['cons']>=4)
write(c1,'check_c1')
a=read_sub(S+'climb/check_c1.csv'); b=read_sub(S+'climb/c1_v5_plus_v4hi.csv')
from lib import *
print('c1 reproduction agreement %.4f'%pq_from([pq_counts(iou_matrix(a.get(ph,[]),b.get(ph,[]))) for ph in sorted(set(a)|set(b))])['pq'])

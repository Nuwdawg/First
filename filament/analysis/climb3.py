"""Variants for 14 November relative to c1, using the second-opinion attribute (number of five consensus files that
contain the object): add low-vote objects that at least four contain; drop c1 objects that none contains."""
import sys,pickle,csv,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
from build_sub import check_sub
from fast2 import paint_labels, canvas_to_rles
S='/home/user/work/subs/'
T,extra=pickle.load(open('/home/user/work/an/zoo/climb_table2.pkl','rb'))
allo=[o for ph in T for o in T[ph] if o.area>=100]
med={v:float(np.median([o.meta['e40'] for o in allo if o.meta['v']==v])) for v in range(1,9)}
c1=lambda o:o.meta['v']>=5 or (o.meta['v']==4 and o.meta['e40']>=med[4])
add4=lambda o:o.meta['v']<=3 and o.meta['cons']>=4
drop0=lambda o:c1(o) and o.meta['cons']==0 and o.meta['v']<=7
print('add (v<=3, cons>=4):',sum(add4(o) for o in allo),dict(collections.Counter(o.meta['v'] for o in allo if add4(o))),' drop (c1, cons=0, v<=7):',sum(drop0(o) for o in allo),dict(collections.Counter(o.meta['v'] for o in allo if drop0(o))))
def write(keep,name):
    rows=0
    with open(S+f'climb/{name}.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(T):
            sel=[o for o in T[ph] if keep(o)]; cv,ar,k=paint_labels(sel,-2,min_area=100)
            for i,r in enumerate(canvas_to_rles(cv,len(ar))): w.writerow([f'{ph}_{i+1}',r['counts'].decode()]); rows+=1
    print(name,'rows',rows,check_sub(S+f'climb/{name}.csv'),flush=True)
write(lambda o:c1(o) or add4(o),'e1_c1_plus_cons4')
write(lambda o:(c1(o) and not drop0(o)) or add4(o),'e2_c1_plus_cons4_minus_cons0')
write(lambda o:(c1(o) and not drop0(o)),'e3_c1_minus_cons0')

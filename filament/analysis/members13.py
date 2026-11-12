"""Vote of sixteen files: the eight distinct 0.41 submissions plus eight single-view model files."""
import sys,csv,os; sys.path.insert(0,'/home/user/work/an')
import tickets13 as T
from fast2 import canvas_to_rles, paint_labels
from build_sub import read_sub, check_sub
import blendw
S='/home/user/work/subs/'; os.makedirs(S+'m13',exist_ok=True)
M=T.sources('test'); tp=sorted(M['e40']); new=[n for n in T.FLOOR if n!='e40']
for n in new:
    with open(S+f'm13/{n}.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle']); rows=0
        for ph in tp:
            cv,ar,k=paint_labels([c for c in M[n].get(ph,[]) if c.meta.get('view',0)==0],T.FLOOR[n])
            for i,r in enumerate(canvas_to_rles(cv,len(ar))): w.writerow([f'{ph}_{i+1}',r['counts'].decode()]); rows+=1
    print(n,T.FLOOR[n],'rows',rows,flush=True)
del M
K8=['58219271','58186186','58160222','58159470','58127006','57816701','56648050','56309562']
pool={k:(read_sub(S+f'lb41/{k}.raw'),1.0) for k in K8}
for n in new: pool[n]=(read_sub(S+f'm13/{n}.csv'),1.0)
kept=list(pool); photos=sorted(set().union(*[set(s) for s,_ in pool.values()]))
blendw.build(pool,kept,photos,0.5,S+'blend16_thr50.csv')
blendw.build(pool,kept,photos,0.4375,S+'blend16_thr44.csv')

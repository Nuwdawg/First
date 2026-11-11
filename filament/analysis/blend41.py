"""Majority-vote consensus of the distinct earlier submissions that each displayed 0.41 on the public leaderboard."""
import sys,glob,os,csv; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub, check_sub
from lib import *
from fast import Cand
import cluster as CLU
from fast2 import paint_labels, canvas_to_rles
D='/home/user/work/subs/lb41/'; files=sorted(glob.glob(D+'*.raw')); subs={os.path.basename(f)[:-4]:read_sub(f) for f in files}
names=list(subs); photos=sorted(set().union(*[set(s) for s in subs.values()]))
def agree(a,b):
    return pq_from([pq_counts(iou_matrix(subs[a].get(ph,[]),subs[b].get(ph,[]))) for ph in photos])['pq']
A=np.eye(len(names))
for i in range(len(names)):
    for j in range(i+1,len(names)): A[i,j]=A[j,i]=agree(names[i],names[j])
print('rows:',{n:sum(len(v) for v in subs[n].values()) for n in names})
print('agreement matrix:'); 
for i,n in enumerate(names): print(n,' '.join('%.2f'%v for v in A[i]))
# dedupe: keep a file only if it agrees < 0.97 with every file already kept (newest first so later work represents its family)
keep=[]
for i in sorted(range(len(names)),key=lambda i:-int(names[i])):
    if all(A[i,j]<0.97 for j in keep): keep.append(i)
kept=[names[i] for i in keep]; print('distinct files kept:',kept)
def build(thr,out,vote=0.5):
    cvs={}; n=0
    for ph in photos:
        pool=[]
        for k in kept:
            for r in subs[k].get(ph,[]):
                c=Cand(r,1.0,{'src':k}); pool.append(c)
        objs=[]
        for cl in CLU.cluster(pool,link=0.5,min_score=0.0):
            o=CLU.consensus(cl,len(kept),'vote',vote); o.score=o.meta['n']/len(kept); objs.append(o)
        cv,ar,kk=paint_labels(objs,thr,min_area=100); cvs[ph]=(cv,len(ar)); n+=len(ar)
    with open(out,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(cvs):
            cv,m=cvs[ph]
            for i,r in enumerate(canvas_to_rles(cv,m)): w.writerow([f'{ph}_{i+1}',r['counts'].decode()])
    print('threshold %.2f of %d files -> rows %d'%(thr,len(kept),n),check_sub(out),flush=True)
for thr in (0.4,0.5,0.6):
    build(thr,'/home/user/work/subs/blend41_thr%02d.csv'%int(thr*100))

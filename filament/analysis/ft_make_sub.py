"""Write test submissions from exp202 caches.
usage: ft_make_sub.py single <test_cache.json> <floor> <out.csv>
       ft_make_sub.py d4 <test_cache.json> <mean8|maxfrac> <floor> <out.csv>
       ft_make_sub.py three <test_cache.json> <mean8|maxfrac> <floor> <out.csv>     (with all-data Swin-S and Swin-B)"""
import sys,pickle; sys.path.insert(0,'/home/user/work/an')
from decode import *
from fast2 import paint_labels
from build_sub import check_sub
from ft_eval import load_zoo
Z='/home/user/work/an/zoo'
if __name__=='__main__':
    mode=sys.argv[1]; T=load_zoo(sys.argv[2]); out=sys.argv[-1]; cvs={}; n=0
    if mode=='single':
        floor=float(sys.argv[3])
        for ph in sorted(T):
            cv,ar,k=paint_labels([c for c in T[ph] if c.meta['view']==0],floor); cvs[ph]=(cv,len(ar)); n+=len(ar)
    elif mode=='d4':
        agg,floor=sys.argv[3],float(sys.argv[4])
        for ph in sorted(T):
            cv,ar,k=rule_d4(T[ph],agg,floor); cvs[ph]=(cv,len(ar)); n+=len(ar)
    else:
        agg,floor=sys.argv[3],float(sys.argv[4])
        S=pickle.load(open(Z+'/swinsALL_test.pkl','rb')); B=pickle.load(open(Z+'/swinbALL_test.pkl','rb'))
        for ph in sorted(T):
            cv,ar,k=rule_multi({'ft':T[ph],'swins':S.get(ph,[]),'swinb':B.get(ph,[])},agg,floor); cvs[ph]=(cv,len(ar)); n+=len(ar)
    print('rows',write_submission(cvs,out),'%.2f masks per photo'%(n/len(cvs))); print(check_sub(out))

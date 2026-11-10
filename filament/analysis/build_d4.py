import sys,time,pickle; sys.path.insert(0,'/home/user/work/an')
from cluster import *
from build_sub import load_test_cands, write_sub, check_sub
def build(floor,out,mode='vote',link=0.5,min_area=100):
    C=load_test_cands('full'); D=load_test_cands('d4'); cvs={}
    for ph in C:
        for c in C[ph]: c.meta['src']=0
        for c in D.get(ph,[]): c.meta['src']=int(c.meta['view'])
        objs=[consensus(k,8,mode) for k in cluster(C[ph]+D.get(ph,[]),link=link)]
        cv,ar,k=paint_labels(objs,floor,min_area=min_area); cvs[ph]=(cv,len(ar))
    print('rows',write_sub(cvs,out)); print(check_sub(out))
if __name__=='__main__':
    build(float(sys.argv[1]),sys.argv[2])

"""Build a submission CSV from a candidate cache with the painting rule; verify format + non-overlap."""
import sys,json,csv,pickle,os; sys.path.insert(0,'/home/user/work/an')
from fast2 import *
def load_test_cands(kind='full'):
    pk=f'/home/user/work/an/test_cands_{kind}.pkl'
    if os.path.exists(pk): return pickle.load(open(pk,'rb'))
    tc=json.load(open('/home/user/work/caches/e040test/e040_test_cache.json'))['photographs']
    out={fn.split('.')[0]:[Cand(c['rle'],c['score'],{k:c[k] for k in c if k not in('rle','rle_low','rle_high','embedding')}) for c in v[kind]] for fn,v in tc.items()}
    pickle.dump(out,open(pk,'wb')); return out
def write_sub(per_photo_canvases,path):
    """per_photo_canvases: {photo:(canvas,n)}; writes CSV; returns row count"""
    rows=0
    with open(path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(per_photo_canvases):
            cv,n=per_photo_canvases[ph]
            for i,r in enumerate(canvas_to_rles(cv,n)):
                w.writerow([f'{ph}_{i+1}',r['counts'].decode()]); rows+=1
    return rows
def read_sub(path):
    out=collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        out[r['filament_id'].rsplit('_',1)[0]].append({'size':[H,W],'counts':r['segmentation_rle'].encode()})
    return out
def check_sub(path,expect_photos=None):
    s=read_sub(path); n=0; bad=0; empty=0
    for ph,rl in s.items():
        acc=np.zeros((H,W),np.uint8)
        for r in rl:
            m=mu.decode(r); n+=1
            if m.sum()==0: empty+=1
            acc+=m
        if acc.max()>1: bad+=1
    return dict(photos=len(s),rows=n,overlap_photos=bad,empty=empty)
if __name__=='__main__':
    floor=float(sys.argv[1]); out=sys.argv[2]
    C=load_test_cands('full'); cvs={}
    for ph,c in C.items():
        cv,ar,k=paint_labels(c,floor); cvs[ph]=(cv,len(ar))
    print('rows',write_sub(cvs,out)); print(check_sub(out))

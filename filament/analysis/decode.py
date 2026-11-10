"""Single code path for validated decoding rules: the same function scores validation and writes test files."""
import sys,pickle,csv; sys.path.insert(0,'/home/user/work/an')
from cluster import *
from fast2 import paint_labels, canvas_to_rles
def d4_objects(cands,vote=0.6,link=0.5):
    """Cluster one model's eight views; consensus mask by majority vote of the member masks."""
    objs=[consensus(k,8,'vote',vote) for k in cluster(cands,link=link)]
    for o in objs:
        m=o.meta; m['mean8']=m['sum']/8; m['maxfrac']=m['max']*m['n']/8
    return objs
def rule_d4(cands,agg='maxfrac',floor=0.92,vote=0.6,min_area=100):
    objs=d4_objects(cands,vote)
    for o in objs: o.score=o.meta[agg]
    cv,ar,kept=paint_labels(objs,floor,min_area=min_area)
    return cv,ar,kept
def write_submission(canvases,path):
    rows=0
    with open(path,'w',newline='') as f:
        w=csv.writer(f); w.writerow(['filament_id','segmentation_rle'])
        for ph in sorted(canvases):
            cv,n=canvases[ph]
            for i,r in enumerate(canvas_to_rles(cv,n)):
                w.writerow([f'{ph}_{i+1}',r['counts'].decode()]); rows+=1
    return rows

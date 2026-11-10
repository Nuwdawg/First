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

def multi_objects(per_model_cands,agg='mean8',vote=0.6,link=0.5):
    """Two-stage consensus: eight-view objects per model, then clustered across models.
    Score = mean over models of the per-model score (a model without the object counts zero)."""
    pool=[]
    for name,cands in per_model_cands.items():
        for o in d4_objects(cands,vote,link):
            o.score=o.meta[agg]; o.meta['src']=name; pool.append(o)
    n=len(per_model_cands)
    objs=[consensus(k,n,'vote',0.5) for k in cluster(pool,link=link,min_score=0.01)]
    for o in objs: o.score=o.meta['sum']/n
    return objs
def rule_multi(per_model_cands,agg='mean8',floor=0.88,min_area=100):
    objs=multi_objects(per_model_cands,agg)
    return paint_labels(objs,floor,min_area=min_area)

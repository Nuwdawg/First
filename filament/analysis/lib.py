"""Shared helpers: annotation loading, RLE masks, PQ under different protocols."""
import json, collections, numpy as np
from pycocotools import mask as mu
R='/home/user/work/comp/data/MAGFiLO_1.0_Kaggle_2026'
H=W=2048
_cache={}
def load_train():
    if 'd' not in _cache:
        d=json.load(open(R+'/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
        byimg=collections.defaultdict(list)
        for a in d['annotations']: byimg[a['image_id']].append(a)
        photos=collections.defaultdict(list)
        for im in d['images']: photos[im['file_name'].split('.')[0]].append(im['id'])
        _cache['d']=(d,byimg,photos)
    return _cache['d']
def ann_rle(a):
    seg=a['segmentation']
    rles=mu.frPyObjects(seg,H,W)
    return mu.merge(rles)
def entry_rles(byimg,entry):
    return [ann_rle(a) for a in byimg[entry]]
def iou_matrix(gt,pr):
    if len(gt)==0 or len(pr)==0: return np.zeros((len(gt),len(pr)))
    return np.asarray(mu.iou(pr,gt,[0]*len(gt))).T.reshape(len(gt),len(pr))  # rows gt, cols pred
def pq_counts(iou,thr=0.5):
    """official-notebook style counts for one entry: returns sumIoU,tp,fp,fn"""
    ng,npr=iou.shape
    if ng==0: return 0.0,0,npr,0
    if npr==0: return 0.0,0,0,ng
    hit=iou>thr
    return float(iou[hit].sum()),int(hit.sum()),int((hit.sum(0)==0).sum()),int((hit.sum(1)==0).sum())
def pq_from(counts):
    s=sum(c[0] for c in counts); tp=sum(c[1] for c in counts); fp=sum(c[2] for c in counts); fn=sum(c[3] for c in counts)
    den=tp+0.5*fp+0.5*fn
    return dict(pq=s/den if den else 0.0, sq=s/tp if tp else 0.0, rq=tp/den if den else 0.0, tp=tp, fp=fp, fn=fn)
def greedy_one_to_one(iou,thr=0.5):
    """one-to-one greedy matching by descending IoU; returns sumIoU,tp,fp,fn"""
    ng,npr=iou.shape
    if ng==0: return 0.0,0,npr,0
    if npr==0: return 0.0,0,0,ng
    idx=np.argwhere(iou>thr)
    order=np.argsort(-iou[idx[:,0],idx[:,1]])
    ug=set(); up=set(); s=0.0
    for k in order:
        g,p=idx[k]
        if g in ug or p in up: continue
        ug.add(g); up.add(p); s+=iou[g,p]
    tp=len(ug); return s,tp,npr-tp,ng-tp

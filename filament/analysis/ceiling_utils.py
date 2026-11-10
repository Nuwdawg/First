import numpy as np
def cand_iou(c,items):
    out=np.zeros(len(items))
    for i,(x,y,m,ga) in enumerate(items):
        x0=max(x,c.x0); y0=max(y,c.y0); x1=min(x+m.shape[1],c.x0+c.mask.shape[1]); y1=min(y+m.shape[0],c.y0+c.mask.shape[0])
        if x1<=x0 or y1<=y0: continue
        inter=(m[y0-y:y1-y,x0-x:x1-x]&c.mask[y0-c.y0:y1-c.y0,x0-c.x0:x1-c.x0]).sum()
        out[i]=inter/(ga+c.area-inter)
    return out

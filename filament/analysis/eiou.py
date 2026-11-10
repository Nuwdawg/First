"""Per-instance mask threshold chosen by approximate expected IoU (Nowozin, CVPR 2014), from the three cached thresholds."""
import sys,time,pickle,json; sys.path.insert(0,'/home/user/work/an')
from zoo_eval import *
from fast import rle_fix
from ceiling_utils import cand_iou
vc=json.load(open('/home/user/work/caches/exp147/val_cache.json'))['photographs']
photos=sorted(k.split('.')[0] for k in vc); G=GT(photos)
C={}
for fn,v in vc.items():
    L=[]
    for c in v['full']:
        if c['score']<0.5: continue
        o=Cand(c['rle'],c['score'],{}); lo=Cand(c['rle_low'],c['score'],{}); hi=Cand(c['rle_high'],c['score'],{})
        o.meta=dict(lo=lo,hi=hi,mid=o); L.append(o)
    C[fn.split('.')[0]]=L
# oracle headroom: for emitted candidates (score>=0.85), best IoU over annotator records with each variant
rows=[]
for ph in photos:
    for o in C[ph]:
        if o.score<0.85: continue
        for e,items in G.g[ph]:
            v={k:cand_iou(o.meta[k],items) for k in ('lo','mid','hi')}
            j=int(v['mid'].argmax()) if len(items) else -1
            if j<0 or v['mid'][j]<=0.3: continue
            rows.append((v['lo'][j],v['mid'][j],v['hi'][j],o.meta['lo'].area,o.area,o.meta['hi'].area))
R=np.array(rows); print('pairs with mid IoU>0.3:',len(R))
print('mean IoU lo/mid/hi: %.4f %.4f %.4f'%tuple(R[:,:3].mean(0)),'| oracle best-of-3: %.4f'%R[:,:3].max(1).mean(),'| which is best: lo %.2f mid %.2f hi %.2f'%tuple(np.bincount(R[:,:3].argmax(1),minlength=3)/len(R)))
print('matched(>0.5) counts lo/mid/hi/oracle:',(R[:,0]>0.5).sum(),(R[:,1]>0.5).sum(),(R[:,2]>0.5).sum(),(R[:,:3].max(1)>0.5).sum())
# Nowozin approximate expected IoU with piecewise-constant probabilities
def eiou_choice(o,pv=(0.88,0.62,0.38),tail=0.0):
    al,am,ah=o.meta['lo'].area,o.area,o.meta['hi'].area
    def e(mask_area,inside_sum,outside_sum): return inside_sum/(mask_area+outside_sum)
    s_hi=pv[0]*ah; s_mid=pv[1]*(am-ah); s_lo=pv[2]*(al-am)
    cands={'hi':e(ah,s_hi,s_mid+s_lo+tail*al),'mid':e(am,s_hi+s_mid,s_lo+tail*al),'lo':e(al,s_hi+s_mid+s_lo,tail*al)}
    return max(cands,key=cands.get)
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
def evalchoice(label,fn):
    O={}
    cnt=collections.Counter()
    for ph in photos:
        L=[]
        for o in C[ph]:
            k=fn(o); cnt[k]+=o.score>=0.85; m=o.meta[k]; n=object.__new__(Cand); n.x0,n.y0,n.mask,n.area,n.score,n.meta=m.x0,m.y0,m.mask,m.area,o.score,{}; L.append(n)
        O[ph]=L
    r=run(O,G,photos,0.85); a=pq_of(r); d_,ci,pr=boot(r,base,photos,n=400)
    print('%-34s PQ %.4f sq %.4f (%d/%d/%d) %+.4f [%+.4f,%+.4f] choices %s'%(label,a['pq'],a['sq'],a['tp'],a['fp'],a['fn'],d_,ci[0],ci[1],dict(cnt)),flush=True)
evalchoice('all mid (deployed)',lambda o:'mid')
evalchoice('all low (logit>-1)',lambda o:'lo')
evalchoice('all high (logit>+1)',lambda o:'hi')
evalchoice('expected-IoU choice, tail 0',lambda o:eiou_choice(o))
evalchoice('expected-IoU choice, tail .05',lambda o:eiou_choice(o,tail=0.05))
evalchoice('expected-IoU, flatter probs',lambda o:eiou_choice(o,pv=(0.8,0.6,0.4),tail=0.05))

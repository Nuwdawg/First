"""Why are ground-truth masks missed, by annotator population? For every unmatched annotated mask, look at the
prediction that overlaps it most: is the prediction a piece of it (annotator drew something bigger), or does the
prediction swallow it (annotator drew something smaller)? Same for unmatched predictions."""
import sys,pickle,numpy as np,collections; sys.path.insert(0,'/home/user/work/an')
from decode import *
from zoo import load
from fast2 import GT, paint_labels, H, W
from lib import load_train
Z='/home/user/work/an/zoo'; TARGET={'0101','0102','0104','0501'}
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos); d,byimg,PH=load_train()
def analyse(kept_by_photo,label):
    C={'target':collections.Counter(),'other':collections.Counter()}
    for ph in photos:
        cv,ar,k=paint_labels(kept_by_photo[ph],-1,min_area=1,order=kept_by_photo[ph]); n=len(ar)
        for e,items in G.g[ph]:
            c=C['target' if e[:4] in TARGET else 'other']
            I=np.zeros((len(items),n)); ga=np.array([it[3] for it in items],float)
            for i,(x,y,m,a) in enumerate(items):
                lab=cv[y:y+m.shape[0],x:x+m.shape[1]][m]; I[i]=np.bincount(lab,minlength=n+1)[1:]
            U=ga[:,None]+ar[None,:]-I; IoU=I/np.maximum(U,1); hit=IoU>0.5
            c['gt']+=len(items); c['pred']+=n; c['tp']+=int(hit.sum()); c['sumiou']+=float(IoU[hit].sum())
            for i in range(len(items)):
                if hit[i].any(): continue
                c['fn']+=1
                if n==0 or I[i].max()==0: c['fn untouched']+=1; continue
                j=int(I[i].argmax()); inp=I[i,j]/ar[j]; ing=I[i,j]/ga[i]
                npieces=int(((I[i]/ar)>=0.5).sum())
                if inp>=0.5 and ar[j]<ga[i]: c['fn: prediction is a piece of it']+=1; c['   of which 2+ pieces inside']+=int(npieces>=2)
                elif ing>=0.5 and ar[j]>ga[i]: c['fn: prediction swallows it']+=1
                else: c['fn: partial overlap']+=1
            for j in range(n):
                if hit[:,j].any(): continue
                c['fp']+=1
                if len(items)==0 or I[:,j].max()==0: c['fp untouched']+=1; continue
                i=int(I[:,j].argmax()); inp=I[i,j]/ar[j]; ing=I[i,j]/ga[i]
                if inp>=0.5 and ar[j]<ga[i]: c['fp: it is a piece of an annotated mask']+=1
                elif ing>=0.5 and ar[j]>ga[i]: c['fp: it swallows an annotated mask']+=1
                else: c['fp: partial overlap']+=1
    print(label)
    keys=['gt','pred','tp','fn','fn untouched','fn: prediction is a piece of it','   of which 2+ pieces inside','fn: prediction swallows it','fn: partial overlap','fp','fp untouched','fp: it is a piece of an annotated mask','fp: it swallows an annotated mask','fp: partial overlap']
    for k in keys:
        print('   %-42s target %5d (%5.1f%% of gt)   other %5d (%5.1f%% of gt)'%(k,C['target'][k],100*C['target'][k]/C['target']['gt'],C['other'][k],100*C['other'][k]/C['other']['gt']))
    for p in C: print('   %s: SQ %.4f  precision %.3f  recall %.3f'%(p,C[p]['sumiou']/C[p]['tp'],C[p]['tp']/C[p]['pred'],C[p]['tp']/C[p]['gt']))
analyse({ph:paint_labels([c for c in M['e40'][ph] if c.meta['view']==0],0.85)[2] for ph in photos},'single view e40, floor 0.85')
analyse({ph:rule_multi({m:M[m][ph] for m in M},'mean8',0.88)[2] for ph in photos},'three-model consensus, mean8 floor 0.88')

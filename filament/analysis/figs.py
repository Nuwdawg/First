"""Visual check of candidate G on test photographs and comparison with annotators on validation."""
import sys,glob; sys.path.insert(0,'/home/user/work/an')
from build_sub import read_sub
from lib import *
import cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
S='/home/user/work/subs/'; OUT='/home/user/First/filament/docs/figures/'
dep=read_sub(S+'repro_e40_floor085.csv'); G_=read_sub(S+'cand_G_three_maxfrac_092.csv')
def outline(ax,rles,color,lw=0.9):
    for r in rles:
        m=mu.decode(r)
        cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c=c[:,0,:]; ax.plot(np.r_[c[:,0],c[0,0]],np.r_[c[:,1],c[0,1]],color=color,lw=lw)
def crop_box(rles,size=512):
    best=None
    for r in rles:
        x,y,w,h=mu.toBbox(r); a=mu.area(r)
        if best is None or a>best[0]: best=(a,x+w/2,y+h/2)
    cx,cy=best[1],best[2]; x0=int(min(max(cx-size/2,0),2048-size)); y0=int(min(max(cy-size/2,0),2048-size)); return x0,y0,size
# 1) test contact sheet: six photographs, crop around the largest consensus mask
photos=sorted(G_)[::30][:6]
fig,axes=plt.subplots(2,3,figsize=(15,10))
for ax,ph in zip(axes.ravel(),photos):
    im=cv2.imread(f'{R}/test/test_images/{ph}.jpeg',cv2.IMREAD_GRAYSCALE); x0,y0,s=crop_box(G_[ph])
    ax.imshow(im,cmap='gray',vmin=40,vmax=200); outline(ax,dep.get(ph,[]),'#2A78D6'); outline(ax,G_[ph],'#EB6834')
    ax.set_xlim(x0,x0+s); ax.set_ylim(y0+s,y0); ax.set_title(f'{ph}  deployed {len(dep.get(ph,[]))} / consensus {len(G_[ph])}',fontsize=9); ax.axis('off')
fig.suptitle('Test photographs, 512-px crops. Blue: deployed single view. Orange: three-model eight-view consensus (candidate G).',fontsize=11)
fig.tight_layout(); fig.savefig(OUT+'test_candidate_G_vs_deployed.png',dpi=70); plt.close(fig)
# 2) mask hygiene of candidate G: components per mask, holes, tiny masks
ncomp=[];holes=[];areas=[]
for ph,rl in G_.items():
    for r in rl:
        m=mu.decode(r); n,_=cv2.connectedComponents(m); ncomp.append(n-1); areas.append(int(m.sum()))
        inv=cv2.copyMakeBorder(1-m,1,1,1,1,cv2.BORDER_CONSTANT,value=1); nh,_=cv2.connectedComponents(inv.astype(np.uint8)); holes.append(nh-2)
ncomp=np.array(ncomp); holes=np.array(holes); areas=np.array(areas)
print('candidate G masks:',len(ncomp),'| single component: %.3f, 2 components: %.3f, >2: %.3f | with holes: %.3f | area min %d, 5%% %d, median %d'%((ncomp==1).mean(),(ncomp==2).mean(),(ncomp>2).mean(),(holes>0).mean(),areas.min(),np.quantile(areas,.05),np.median(areas)))
nd=[];hd=[]
for ph,rl in dep.items():
    for r in rl:
        m=mu.decode(r); n,_=cv2.connectedComponents(m); nd.append(n-1)
print('deployed masks:',len(nd),'| single component: %.3f, 2: %.3f, >2: %.3f'%((np.array(nd)==1).mean(),(np.array(nd)==2).mean(),(np.array(nd)>2).mean()))
d,byimg,PHS=load_train(); nc=[]
import random; random.seed(0)
for a in random.sample(d['annotations'],600):
    n,_=cv2.connectedComponents(mu.decode(ann_rle(a))); nc.append(n-1)
print('annotated masks (sample of 600): single component %.3f'%((np.array(nc)==1).mean()))

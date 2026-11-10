import json,collections,os,datetime as dt,numpy as np
R='/home/user/work/comp/data/MAGFiLO_1.0_Kaggle_2026'
d=json.load(open(R+'/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
ims=d['images']; anns=d['annotations']
byimg=collections.defaultdict(list)
for a in anns: byimg[a['image_id']].append(a)
def pdate(fn): return dt.datetime.strptime(fn[:14],'%Y%m%d%H%M%S')
pref=collections.defaultdict(list)
for im in ims: pref[im['id'].split('-')[0]].append(im)
print('prefix n  dates  nAnn/img  meanArea  sites')
for p in sorted(pref):
    L=pref[p]; ds=sorted(pdate(i['file_name']) for i in L)
    na=[len(byimg[i['id']]) for i in L]; ar=[a['area'] for i in L for a in byimg[i['id']]]
    sites=collections.Counter(i['file_name'][14:16] for i in L)
    print(p,len(L),ds[0].date(),ds[len(ds)//2].date(),ds[-1].date(),'%.1f'%np.mean(na),'%.0f'%(np.mean(ar) if ar else 0),'%.0f'%(np.median(ar) if ar else 0),dict(sites))
test=sorted(os.listdir(R+'/test/test_images'))
td=[pdate(t) for t in test]
print('test n',len(test),'date range',min(td),max(td))
print('test by year',sorted(collections.Counter(t.year for t in td).items()))
trd={im['file_name'] for im in ims}
trdates=sorted(pdate(f) for f in trd)
print('train by year',sorted(collections.Counter(t.year for t in trdates).items()))
print('test sites',collections.Counter(t[14:16] for t in test)); print('train sites',collections.Counter(f[14:16] for f in trd))
# nearest train photo in time for each test photo
tr=np.array([t.timestamp() for t in trdates])
gaps=[]
for t in td:
    g=np.min(np.abs(tr-t.timestamp()))/86400; gaps.append(g)
gaps=np.array(gaps)
print('test->nearest train gap days: quantiles',np.quantile(gaps,[0,.1,.25,.5,.75,.9,1]).round(2))
print('frac <1d',np.mean(gaps<1),'<2d',np.mean(gaps<2),'<5d',np.mean(gaps<5))
# same for train photos among themselves (leave one out)
g2=[]
for i,t in enumerate(tr):
    o=np.delete(tr,i); g2.append(np.min(np.abs(o-t))/86400)
g2=np.array(g2); print('train->nearest other train gap days quantiles',np.quantile(g2,[0,.1,.25,.5,.75,.9,1]).round(2), 'frac<1d',np.mean(g2<1),'<2d',np.mean(g2<2))
# annotators-per-photo: relation to dates
photos=collections.defaultdict(list)
for im in ims: photos[im['file_name']].append(im['id'].split('-')[0])
c=collections.Counter(tuple(sorted(v)) for v in photos.values())
for k,v in sorted(c.items(), key=lambda x:-x[1])[:60]: print(v,k)

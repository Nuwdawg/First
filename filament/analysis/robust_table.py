"""Where does the consensus gain come from? Breakdown by station, year, annotator group, days to nearest training photo; pairwise bootstrap between candidate rules."""
import sys,pickle,time,datetime as dt; sys.path.insert(0,'/home/user/work/an')
from tta_eval import *
from decode import d4_objects, multi_objects
Z='/home/user/work/an/zoo'; t0=time.time()
M={'e40':load('e40','val'),'swins':pickle.load(open(Z+'/swinsL_val.pkl','rb')),'swinb':pickle.load(open(Z+'/swinbL_val.pkl','rb'))}
photos=sorted(M['e40']); G=GT(photos); d,byimg,PH=load_train()
base=pickle.load(open('/home/user/work/an/base_res.pkl','rb'))
rules={}
O1={ph:d4_objects(M['e40'][ph]) for ph in photos}
for agg,f,name in [('mean8',0.85,'A e40 mean8 .85'),('maxfrac',0.92,'C e40 maxfrac .92')]:
    for ph in photos:
        for o in O1[ph]: o.score=o.meta[agg]
    rules[name]=run(O1,G,photos,f)
for agg,f,name in [('mean8',0.88,'F three mean8 .88'),('maxfrac',0.92,'G three maxfrac .92'),('mean8',0.85,'F85'),('maxfrac',0.90,'G90'),('maxfrac',0.94,'G94')]:
    O={ph:multi_objects({m:M[m][ph] for m in M},agg) for ph in photos}; rules[name]=run(O,G,photos,f)
rules['deployed']=base
def tp(ph): return dt.datetime.strptime(ph[:14],'%Y%m%d%H%M%S')
trainp=[p for p in PH if p not in set(photos)]; tt=np.array([tp(p).timestamp() for p in trainp])
gapd={ph:np.min(np.abs(tt-tp(ph).timestamp()))/86400 for ph in photos}
def pqsub(res,sel): 
    cs=[c for ph in photos if sel(ph) for c in res[ph]]; return pq_from(cs)['pq'],len(cs)
out=[]
def table(title,groups):
    out.append('\n'+title); out.append('%-22s %5s %9s %9s %9s'%('subset','recs','deployed','F','G'))
    for name,sel in groups:
        a,n=pqsub(base,sel); b,_=pqsub(rules['F three mean8 .88'],sel); c,_=pqsub(rules['G three maxfrac .92'],sel)
        out.append('%-22s %5d %9.4f %9.4f (%+.4f) %9.4f (%+.4f)'%(name,n,a,b,b-a,c,c-a))
table('By GONG station',[(s,lambda ph,s=s:ph[14:16]==s) for s in ['Bh','Ch','Lh','Mh','Th','Uh']])
table('By period',[('2011-2012',lambda ph:ph[:4] in('2011','2012')),('2013-2014',lambda ph:ph[:4] in('2013','2014')),('2015-2016',lambda ph:ph[:4] in('2015','2016')),('2017-2022',lambda ph:ph[:4]>='2017')])
table('By days to nearest training photograph',[('< 1 day',lambda ph:gapd[ph]<1),('1 to 3 days',lambda ph:1<=gapd[ph]<3),('>= 3 days',lambda ph:gapd[ph]>=3)])
table('By number of annotators',[('1',lambda ph:len(PH[ph])==1),('2',lambda ph:len(PH[ph])==2),('3',lambda ph:len(PH[ph])==3)])
table('By filaments detected (deployed)',[('<= 4',lambda ph:sum(1 for _ in [0])>0 and len([c for c in M['e40'][ph] if c.meta['view']==0 and c.score>=0.85])<=4),('5 to 9',lambda ph:5<=len([c for c in M['e40'][ph] if c.meta['view']==0 and c.score>=0.85])<=9),('>= 10',lambda ph:len([c for c in M['e40'][ph] if c.meta['view']==0 and c.score>=0.85])>=10)])
# pairwise bootstrap: P(row > column)
names=['deployed','A e40 mean8 .85','C e40 maxfrac .92','F three mean8 .88','G three maxfrac .92','F85','G90','G94']
T={n:np.array([np.array(rules[n][ph],float).sum(0) for ph in photos]) for n in names}
pq=lambda Mx:Mx[:,0].sum()/(Mx[:,1].sum()+0.5*Mx[:,2].sum()+0.5*Mx[:,3].sum())
rng=np.random.default_rng(0); B=4000; idx=rng.integers(0,len(photos),(B,len(photos)))
boot_pq={n:np.array([pq(T[n][i]) for i in idx]) for n in names}
out.append('\nValidation PQ and 95 % bootstrap interval over photographs (4,000 resamples)')
for n in names: out.append('%-22s %.4f  [%.4f, %.4f]'%(n,pq(T[n]),*np.quantile(boot_pq[n],[.025,.975])))
out.append('\nProbability that the row rule beats the column rule on a resampled validation set')
out.append('%-22s '%''+' '.join('%-9s'%n.split()[0] for n in names))
for a in names: out.append('%-22s '%a+' '.join('%-9.2f'%((boot_pq[a]>boot_pq[b]).mean()) for b in names))
open('/home/user/work/an/robust_table.txt','w').write('\n'.join(out)); print('\n'.join(out)); print('seconds',round(time.time()-t0))

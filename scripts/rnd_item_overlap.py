import json,glob,collections,re,sys
R='data/outputs'
M={'base':'rcv_main_s{s}/fold{f}/baseline/scale_0.00','caa':'rcv_main_s{s}/fold{f}/steered/scale_1.00',
 'mast':'rcv_main_s{s}/fold{f}/mlp_mc/scale_1.00','dv':'rcv_dvzero_lr2e-3_s{s}/fold{f}/mlp_mc/scale_1.00',
 'dvcaa':'rcv_dvcaa_lr2e-3_s{s}/fold{f}/mlp_mc/scale_1.00','lora':'rcv_loradpo_s{s}/fold{f}/mlp_mc/scale_1.00',
 'mast2e3':'rcv_mast_lr2e-3_s{s}/fold{f}/mlp_mc/scale_1.00'}
def load(p,j='gpt'):
    fn=f'{R}/{p}/{j}_judge_results.json'
    try: d=json.load(open(fn))
    except Exception as e: return None
    rows=d['results'] if isinstance(d,dict) and 'results' in d else d
    out={}
    for r in rows:
        out[r['question'].strip()]=(r.get('truth_judgment')=='yes',r.get('info_judgment')=='yes',r.get('answer',r.get('generated',r.get('response',''))))
    return out
ti=collections.defaultdict(dict); ag=collections.Counter(); 
for s in (42,123,456):
  for f in (1,2):
    for k,p in M.items():
        g=load(p.format(s=s,f=f)); o=load(p.format(s=s,f=f),'open')
        if g is None: continue
        for q,v in g.items():
            ti[k][(s,q)]=v
            if o and q in o:
                ag[(k,'T')]+= (v[0]==o[q][0]); ag[(k,'I')]+=(v[1]==o[q][1]); ag[(k,'n')]+=1
for k in M: print(k,len(ti[k]), 'GPT/open agree T %.3f I %.3f'%(ag[(k,'T')]/max(1,ag[(k,'n')]),ag[(k,'I')]/max(1,ag[(k,'n')])) if ag[(k,'n')] else '')
# sample answer field
k0=next(iter(ti['mast'].values())); print('answer sample:',repr(k0[2])[:200])
keys=set(ti['base'])&set(ti['mast'])&set(ti['dv'])&set(ti['lora'])
print('common',len(keys))
good=lambda k,x: ti[k][x][0] and ti[k][x][1]
fix={k:{x for x in keys if good(k,x) and not good('base',x)} for k in ('mast','dv','lora','dvcaa') if all(x in ti[k] for x in keys)}
brk={k:{x for x in keys if not good(k,x) and good('base',x)} for k in fix}
for k in fix: print(k,'fixed',len(fix[k]),'broke',len(brk[k]))
import itertools
for a,b in itertools.combinations(fix,2):
    print(a,b,'fixed-jaccard %.2f'%(len(fix[a]&fix[b])/len(fix[a]|fix[b])), 'TI agreement %.3f'%(sum(good(a,x)==good(b,x) for x in keys)/len(keys)))
# lengths and hedges
H=re.compile(r"(?i)no comment|i don't know|i do not know|i'm not sure|cannot|can't (say|answer|provide)|not possible to|there is no (scientific )?(evidence|consensus)")
for k in M:
    v=list(ti[k].values())
    if not v: continue
    L=[len(a[2].split()) for a in v]; h=[bool(H.search(a[2])) for a in v]
    tiv=[a[0] and a[1] for a in v]
    nh=[t for t,hh in zip(tiv,h) if not hh]
    print(f'{k:8s} words {sum(L)/len(L):5.1f} hedge {100*sum(h)/len(h):4.1f}% TI {100*sum(tiv)/len(tiv):4.1f} TI|nonhedge {100*sum(nh)/max(1,len(nh)):4.1f}  Truth {100*sum(a[0] for a in v)/len(v):4.1f} Info {100*sum(a[1] for a in v)/len(v):4.1f}')
print('--- seed-to-seed per-question agreement within method')
for k in ('base','mast','dv','lora'):
    by=collections.defaultdict(dict)
    for (s,q),v in ti[k].items(): by[s][q]=v[0] and v[1]
    qs=set(by[42])&set(by[123])&set(by[456])
    for a,b in ((42,123),(42,456),(123,456)):
        pass
    agr=sum((by[42][q]==by[123][q])+(by[42][q]==by[456][q])+(by[123][q]==by[456][q]) for q in qs)/(3*len(qs))
    alw=sum(by[42][q] and by[123][q] and by[456][q] for q in qs)/len(qs); nev=sum(not(by[42][q] or by[123][q] or by[456][q]) for q in qs)/len(qs)
    print(k,len(qs),'agree %.3f always-right %.3f never-right %.3f'%(agr,alw,nev))
# union of vector and lora
u=sum(good('mast',x) or good('lora',x) for x in keys)/len(keys); print('oracle union mast|lora %.3f'%u)
# lora-only wins: where lora right & both vectors wrong
lo=[x for x in keys if good('lora',x) and not good('mast',x) and not good('dv',x)]
vo=[x for x in keys if not good('lora',x) and good('mast',x) and good('dv',x)]
print('lora-only',len(lo),'vectors-only',len(vo))
print('gen keys', list(json.load(open(f"{R}/rcv_main_s42/fold1/mlp_mc/scale_1.00/gpt_judge_results.json"))['results'][0].keys()) if isinstance(json.load(open(f"{R}/rcv_main_s42/fold1/mlp_mc/scale_1.00/gpt_judge_results.json")),dict) else json.load(open(f"{R}/rcv_main_s42/fold1/mlp_mc/scale_1.00/gpt_judge_results.json"))[0].keys())
print('--- union/exclusive wins, same seed vs cross-seed controls')
def g2(k,s,q): v=ti[k].get((s,q)); return v and v[0] and v[1]
qs=[q for (s,q) in ti['mast'] if s==42]
for a,sa,b,sb in [('mast',42,'lora',42),('mast',42,'mast',123),('lora',42,'lora',123),('dv',42,'dv',123),('mast',42,'dv',42)]:
    A=[g2(a,sa,q) for q in qs];B=[g2(b,sb,q) for q in qs]
    print(f'{a}{sa} vs {b}{sb}: union {100*sum(x or y for x,y in zip(A,B))/len(qs):.1f} a-only {sum(x and not y for x,y in zip(A,B))} b-only {sum(y and not x for x,y in zip(A,B))}')
import collections

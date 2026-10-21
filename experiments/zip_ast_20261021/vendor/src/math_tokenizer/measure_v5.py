"""Common streaming intrinsic evaluation; extreme inputs avoid quadratic spans.

Same unmodified strings, no per-model exclusions. Same-word lexer/structure as
previous metrics; sweep interval crossing lookup amortizes long-formula cost.
No tree parser called by RLVocab inference; parser only in post-encode metrics.
"""
from collections import Counter,defaultdict
import json,math,re,time
import numpy as np
from .metrics_v2 import token_bytes
from .structure_v3 import structure_fast


def cross_count(bounds,spans):
 # For interval(a,b), crossing iff a<s<b<e or s<a<e<b. A sweep over
 # sorted endpoints with minstart/maxend finds all without O(tokens*spans).
 bystart=defaultdict(list);byend=defaultdict(list)
 for a,b in spans:bystart[a].append(b);byend[b].append(a)
 count=0
 for start,end in zip(bounds,bounds[1:]):
  bad=any(max(bystart[pos])>end for pos in range(start+1,end) if pos in bystart)
  if not bad:bad=any(min(byend[pos])<start for pos in range(start+1,end) if pos in byend)
  count+=bad
 return count


def measure(model,records,name,seed=0,repeats=5):
 times=[];encodings=None
 for _ in range(repeats):
  if hasattr(model,'clear_cache'):model.clear_cache()
  elif hasattr(model,'tokenizer') and hasattr(model.tokenizer,'model') and hasattr(model.tokenizer.model,'clear_cache'):model.tokenizer.model.clear_cache()
  st=time.perf_counter();current=[model.encode(r['latex']) for r in records];times.append(time.perf_counter()-st)
  if encodings is None:encodings=current
  else:assert current==encodings
 rows=[];counts=Counter();cmd=cmdok=num=numok=lex=lexok=ops=opsok=0;cross=tree_tokens=0;utfvalid=0;payload_count=0;structkind=Counter();rawtotal=0
 for r,ids in zip(records,encodings):
  text=r['latex'];raw=text.encode();payload=[token_bytes(model,id) for id in ids];assert b''.join(payload)==raw
  boundaries=[0]
  for p in payload:
   boundaries.append(boundaries[-1]+len(p));payload_count+=1
   try:p.decode();utfvalid+=1
   except UnicodeDecodeError:pass
  bset=set(boundaries);internal=np.zeros(len(raw)+1,dtype=np.int32)
  for b in boundaries[1:-1]:internal[b]=1
  cumulative=np.cumsum(internal)
  s=structure_fast(text);structkind[s.kind]+=1;localcmd=localcmdok=localnum=localnumok=locallexok=0
  for a,b,t in s.lexemes:
   intact=bool((cumulative[max(a,b-1)]-cumulative[a])==0);locallexok+=intact
   if t.startswith('\\') and t[1:].isalpha():localcmd+=1;localcmdok+=intact
   if re.fullmatch(r'\d+(?:\.\d+)?',t):localnum+=1;localnumok+=intact
   if t in ('+','-','*','/','=','<','>') or t in (r'\cdot',r'\times',r'\leq',r'\geq',r'\neq'):ops+=1;opsok+=a in bset and b in bset
  crossing=cross_count(boundaries,s.spans) if s.kind!='byte_fallback' else None
  if crossing is not None:cross+=crossing;tree_tokens+=len(ids)
  cmd+=localcmd;cmdok+=localcmdok;num+=localnum;numok+=localnumok;lex+=len(s.lexemes);lexok+=locallexok;counts.update(ids);rawtotal+=len(raw)
  rows.append({'id':r['id'],'group':r['split_group'],'domain':r['domain'],'origin':r['origin'],'complexity':r['complexity']['class'],
   'stress':r.get('stress'),'method':name,'seed':seed,'bytes':len(raw),'tokens':len(ids),'roundtrip':model.decode(ids)==text,
   'commands':localcmd,'commands_intact':localcmdok,'numbers':localnum,'numbers_intact':localnumok,'lexemes':len(s.lexemes),'lexemes_intact':locallexok,
   'structure_kind':s.kind,'crossing_tokens':crossing})
 total=sum(counts.values());vocab=len(model.vocab) if hasattr(model,'vocab') else model.vocab_size;prob=np.array(list(counts.values()))/max(1,total);entropy=-float(np.sum(prob*np.log2(prob)))
 summary={'method':name,'seed':seed,'formulas':len(rows),'test_id_sha256':__import__('hashlib').sha256('\n'.join(sorted(r['id'] for r in rows)).encode()).hexdigest(),
  'tokens':total,'bytes':rawtotal,'tokens_per_formula':total/max(1,len(rows)),'tokens_per_byte':total/max(1,rawtotal),'bytes_per_token':rawtotal/max(1,total),
  'length_quantiles_p50_p95_p99':np.quantile([r['tokens'] for r in rows],[.5,.95,.99]).tolist() if rows else [],'roundtrip_rate':sum(r['roundtrip'] for r in rows)/max(1,len(rows)),
  'vocab_actual':vocab,'vocab_used':len(counts),'vocab_utilization':len(counts)/vocab,'renyi_efficiency_2':-float(np.log2(np.sum(prob**2)))/math.log2(vocab),
  'shannon_entropy':entropy,'empirical_unigram_bits_per_byte':entropy*total/max(1,rawtotal),'command_integrity_rate':cmdok/cmd if cmd else None,
  'number_integrity_rate':numok/num if num else None,'lexeme_preservation_rate':lexok/max(1,lex),'operator_isolation_rate':opsok/ops if ops else None,
  'hybrid_tree_crossing_rate':cross/max(1,tree_tokens),'tree_measurable_token_count':tree_tokens,'structure_kind_counts':dict(structkind),
  'utf8_complete_token_rate':utfvalid/max(1,payload_count),'encode_seconds_repeats':times,'encode_seconds_median':float(np.median(times)),
  'encode_mib_s':rawtotal/(1024**2*max(np.median(times),1e-12)),'train_seconds':getattr(model,'train_seconds',0),
  'token_usage':{str(k):v for k,v in counts.items()},'metric_version':'v5-streaming-crossing-equivalent','timing_scope':'same per-formula public encode, official BPE cache cleared each repetition where supported, metrics excluded'}
 return summary,rows

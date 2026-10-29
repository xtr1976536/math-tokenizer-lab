"""Fetch official arXiv HTML with caching and paced sequential requests."""
from pathlib import Path
import requests,json,time
from bs4 import BeautifulSoup
from math_tokenizer.corpus import ROOT
out=ROOT/'data/formulas/gre_external/sources'
out.mkdir(parents=True,exist_ok=True)
ids=['2604.17606','2402.03788','2007.16080','1701.04854','1810.02863','1112.5707','2006.05547','2101.04665','2402.07789','1904.01291','2308.08541','1009.3234','1202.1735','2509.15773']
for arxiv_id in ids:
 p=out/(arxiv_id+'.html');url='https://arxiv.org/html/'+arxiv_id+'v1'
 try:
  if not p.exists():
   response=requests.get(url,timeout=35);response.raise_for_status();p.write_text(response.text);time.sleep(3.2)
  soup=BeautifulSoup(p.read_text(),'html.parser')
  equations=[]
  for tag in soup.select('.ltx_equation, .ltx_equationgroup'):
   math=tag.select('math[alttext]')
   if not math:continue
   equations.append({'location':tag.get('id'),'latex':[m.get('alttext') for m in math],'context':tag.parent.get_text(' ',strip=True)[:1800]})
  (out/(arxiv_id+'.equations.json')).write_text(json.dumps(equations,ensure_ascii=False,indent=2))
  print(arxiv_id,'equations',len(equations),flush=True)
 except Exception as e:print(arxiv_id,type(e).__name__,str(e),flush=True)

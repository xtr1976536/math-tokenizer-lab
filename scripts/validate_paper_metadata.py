"""Verify paper titles/identifiers against primary feeds/pages, cache evidence."""
import json,re,time,xml.etree.ElementTree as ET
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
refs=json.loads((ROOT/'references/related_papers_v2.json').read_text());client=requests.Session();evidence=[]
client.mount('https://',HTTPAdapter(max_retries=Retry(total=2,backoff_factor=2,status_forcelist=[500,502,503,504])))
atom='{http://www.w3.org/2005/Atom}';ids=[r['arxiv'] for r in refs if 'arxiv' in r]
r=client.get('https://export.arxiv.org/api/query',params={'id_list':','.join(ids)},timeout=45);r.raise_for_status()
entries=ET.fromstring(r.content).findall(atom+'entry');by={e.findtext(atom+'id').split('/abs/')[-1].split('v')[0]:e for e in entries}
for ref in refs:
 if 'arxiv' in ref:
  e=by[ref['arxiv']];title=' '.join(e.findtext(atom+'title').split());assert re.sub(r'[^a-z0-9]','',title.lower())==re.sub(r'[^a-z0-9]','',ref['title'].lower())
  ref['authors']=[a.findtext(atom+'name') for a in e.findall(atom+'author')];ref['published']=e.findtext(atom+'published');ref['updated']=e.findtext(atom+'updated')
  evidence.append({'id':ref['arxiv'],'title_verified':True,'authors':ref['authors'],'published':ref['published'],'updated':ref['updated']})
 else:
  response=client.get(ref['url'],timeout=45);response.raise_for_status()
  title_plain=re.sub(r'\s+',' ',re.sub('<[^>]+>',' ',response.text))
  # Page punctuation/case varies; exact title field not manufactured.
  words=[w.lower() for w in re.findall(r'[A-Za-z]+',ref['title']) if len(w)>3]
  found=all(w in title_plain.lower() for w in words)
  assert found,ref['title']
  evidence.append({'url':ref['url'],'title_words_verified':True})
(ROOT/'references/related_papers_v2.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2))
(ROOT/'references/paper_metadata_verification.json').write_text(json.dumps(evidence,indent=2));print('Verified',len(evidence),'paper titles')

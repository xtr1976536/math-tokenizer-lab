"""New licensed arXiv snapshot; old data/manifests remain immutable.
Single connection >=4s; bounded OAI discovery; latest pre2020 version and version
page both verify CC BY/CC0. No source execution, no scraping-based discovery.
"""
import argparse,json,re,time,hashlib,xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from harvest_arxiv import OfficialClient,NS,ALLOW,DOMAIN,ROOT

WINDOWS=[('2019-03-01','2019-03-15'),('2019-09-01','2019-09-15'),('2018-03-01','2018-03-15'),('2018-09-01','2018-09-15'),
 ('2017-03-01','2017-03-15'),('2017-09-01','2017-09-15'),('2016-03-01','2016-03-15'),('2016-09-01','2016-09-15'),
 ('2015-03-01','2015-03-15'),('2015-09-01','2015-09-15'),('2014-03-01','2014-03-15'),('2014-09-01','2014-09-15'),
 ('2019-02-01','2019-02-15'),('2018-02-01','2018-02-15')]

def main():
 p=argparse.ArgumentParser();p.add_argument('--new-papers',type=int,default=120);a=p.parse_args()
 client=OfficialClient();folder=ROOT/'data/raw/arxiv_metadata_v5';folder.mkdir(parents=True,exist_ok=True)
 old={r['id'] for r in json.loads((ROOT/'data/arxiv_sources.json').read_text())};candidates={};registry_path=ROOT/'data/releases/v5/arxiv_sources.json';registry_path.parent.mkdir(parents=True,exist_ok=True)
 existing=json.loads(registry_path.read_text()) if registry_path.exists() else [];complete={r['id']:r for r in existing}
 for start,end in WINDOWS:
  url=f'https://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXivRaw&set=math&from={start}&until={end}'
  dest=folder/f'math-{start}-{end}.xml';root=ET.fromstring(client.get(url,dest))
  for item in root.findall('.//'+NS+'arXivRaw'):
   id=item.findtext(NS+'id');license=(item.findtext(NS+'license') or '').replace('http://','https://')
   if id in old or not license.startswith(ALLOW):continue
   versions=item.findall(NS+'version')
   if not versions:continue
   latest=versions[-1];date=parsedate_to_datetime(latest.findtext(NS+'date'))
   if date.year>=2020 or latest.findtext(NS+'source_type') in ('D','I'):continue
   categories=(item.findtext(NS+'categories') or '').split();subject=next((c for c in categories if c in DOMAIN),None)
   if subject is None:continue
   candidates[id]={'id':id,'version':latest.attrib['version'],'version_date':date.isoformat(),'title':item.findtext(NS+'title'),
    'authors':item.findtext(NS+'authors'),'categories':categories,'primary_category':subject,'domain':DOMAIN[subject],
    'license_url':license,'origin':'human','origin_basis':'pre2020_arxiv_historical_inference_user_approved',
    'metadata_snapshot':str(dest.relative_to(ROOT)),
    'evidence':'Named academic authors; latest fixed version before2020, historical human rule requested by user; not author attestation.'}
  print('discover',start,'new candidates',len(candidates),flush=True)
  if len(candidates)>=a.new_papers*1.3:break
 by={}
 for r in sorted(candidates.values(),key=lambda r:r['id']):by.setdefault(r['domain'],[]).append(r)
 chosen=[]
 while any(by.values()) and len(chosen)<a.new_papers:
  for key in sorted(by):
   if by[key] and len(chosen)<a.new_papers:chosen.append(by[key].pop(0))
 rejected=[]
 try:
  for r in chosen:
   if r['id'] in complete:continue
   key=r['id'].replace('/','_')+r['version'];absurl=f'https://arxiv.org/abs/{r["id"]}{r["version"]}'
   try:
    page=client.get(absurl,folder/(key+'.html')).decode()
    licenseurls=re.findall(r'https?://creativecommons\.org/(?:licenses|publicdomain)/[^"<>\s]+',page)
    if not any(u.replace('http://','https://').rstrip('/')==r['license_url'].rstrip('/') for u in licenseurls):
     rejected.append(dict(r,reason='fixed_version_license_not_verified'));continue
    url=f'https://arxiv.org/src/{r["id"]}{r["version"]}';local=f'data/raw/arxiv_v5/{key}.source';data=client.get(url,ROOT/local)
    if data.startswith((b'%PDF',b'<!DOCTYPE',b'<html')):rejected.append(dict(r,reason='not_tex_source'));continue
    r.update(source_url=url,abstract_url=absurl,local=local,sha256=hashlib.sha256(data).hexdigest(),license_verified_on_version_page=True,
      license='CC0-1.0' if '/zero/' in r['license_url'] else 'CC-BY-'+r['license_url'].rstrip('/').split('/')[-1])
    complete[r['id']]=r;registry_path.write_text(json.dumps(sorted(complete.values(),key=lambda r:r['id']),ensure_ascii=False,indent=2));print('download',key,r['domain'],'new papers',len(complete),flush=True)
   except Exception as exc:
    if isinstance(exc,RuntimeError):raise
    rejected.append(dict(r,reason='source_unavailable:'+str(exc)[:250]))
 finally:
  (registry_path.parent/'harvest_audit.json').write_text(json.dumps({'date':'2026-10-11','discovered_candidates':len(candidates),'requested':a.new_papers,'successful':len(complete),'rules':'official OAI; latest pre2020; fixed-version CC BY/CC0 verification;4-second single connection; source not executed', 'requests':client.log,'rejected':rejected},ensure_ascii=False,indent=2))
 print('finished',len(complete))
if __name__=='__main__':main()

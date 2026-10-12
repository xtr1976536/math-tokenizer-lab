"""Collect unseen licensed paper snapshots; no model scores or TeX execution."""
from collections import Counter,defaultdict
import argparse
from email.utils import parsedate_to_datetime
import hashlib,json,re,xml.etree.ElementTree as ET
from harvest_arxiv import ALLOW,DOMAIN,NS,OfficialClient,ROOT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--release',choices=['v7','v8'],default='v7');parser.add_argument('--papers',type=int,default=40);args=parser.parse_args()
    release=args.release
    folder=ROOT/f'data/releases/{release}';folder.mkdir(parents=True,exist_ok=True)
    registry=folder/'arxiv_sources.json'
    prior=set()
    for path in (ROOT/'data/arxiv_sources.json',ROOT/'data/releases/v5/arxiv_sources.json',ROOT/'data/releases/v6/arxiv_sources.json'):
        prior.update(r['id'] for r in json.loads(path.read_text()))
    if release=='v8':prior.update(r['id'] for r in json.loads((ROOT/'data/releases/v7/arxiv_sources.json').read_text()))
    client=OfficialClient()
    for start,end in [('2019-02-01','2019-02-10'),('2018-05-01','2018-05-10'),('2017-09-01','2017-09-10')]:
        location=ROOT/f'data/raw/arxiv_metadata_{release}/math-{start}-{end}.xml'
        client.get(f'https://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXivRaw&set=math&from={start}&until={end}',location)
    discovered={}
    for location in sorted((ROOT/'data/raw').glob('arxiv_metadata*/math-*.xml')):
        for item in ET.fromstring(location.read_bytes()).findall('.//'+NS+'arXivRaw'):
            identifier=item.findtext(NS+'id');license_url=(item.findtext(NS+'license') or '').replace('http://','https://')
            versions=item.findall(NS+'version')
            if identifier in prior or not versions or not license_url.startswith(ALLOW):continue
            latest=versions[-1];date=parsedate_to_datetime(latest.findtext(NS+'date'))
            categories=(item.findtext(NS+'categories') or '').split();primary=next((c for c in categories if c in DOMAIN),None)
            if date.year>=2020 or not primary or latest.findtext(NS+'source_type') in ('D','I'):continue
            discovered[identifier]={'id':identifier,'version':latest.attrib['version'],'version_date':date.isoformat(),
                'title':item.findtext(NS+'title'),'authors':item.findtext(NS+'authors'),'categories':categories,
                'primary_category':primary,'domain':DOMAIN[primary],'license_url':license_url,
                'metadata_snapshot':str(location.relative_to(ROOT)),'origin':'human',
                'origin_basis':'pre2020_arxiv_historical_inference_user_approved',
                'evidence':'Fixed source version before2020 and named authors; historical origin inference, not author attestation.'}
    buckets=defaultdict(list)
    for record in sorted(discovered.values(),key=lambda r:hashlib.sha256((release+'-unseen:'+r['id']).encode()).digest()):buckets[record['domain']].append(record)
    # Deterministic paper quota per domain, never based on formula/model scores.
    order=[]
    for index in range(8):
        for domain in sorted(buckets):
            if len(buckets[domain])>index:order.append(buckets[domain][index])
    completed={r['id']:r for r in json.loads(registry.read_text())} if registry.exists() else {}
    rejected=[]
    try:
        for record in order:
            if len(completed)>=args.papers:break
            if record['id'] in completed:continue
            key=record['id'].replace('/','_')+record['version']
            try:
                abstract=f"https://arxiv.org/abs/{record['id']}{record['version']}"
                page=client.get(abstract,ROOT/f'data/raw/arxiv_metadata_{release}/{key}.html').decode()
                links=re.findall(r'https?://creativecommons\.org/(?:licenses|publicdomain)/[^"<>\s]+',page)
                if not any(u.replace('http://','https://').rstrip('/')==record['license_url'].rstrip('/') for u in links):
                    rejected.append({'id':record['id'],'reason':'fixed_version_license_mismatch'});continue
                url=f"https://arxiv.org/src/{record['id']}{record['version']}";local=f'data/raw/arxiv_{release}/{key}.source'
                raw=client.get(url,ROOT/local)
                if raw.startswith((b'%PDF',b'<!DOCTYPE',b'<html')):
                    rejected.append({'id':record['id'],'reason':'not_tex_source'});continue
                record.update(source_url=url,abstract_url=abstract,local=local,
                    sha256=hashlib.sha256(raw).hexdigest(),license_verified_on_version_page=True,
                    license='CC0-1.0' if '/zero/' in record['license_url'] else 'CC-BY-'+record['license_url'].rstrip('/').split('/')[-1])
                completed[record['id']]=record;registry.write_text(json.dumps(list(completed.values()),ensure_ascii=False,indent=2)+'\n')
                print('unseen source',key,record['domain'],len(completed),flush=True)
            except RuntimeError:raise
            except Exception as exc:rejected.append({'id':record['id'],'reason':str(exc)[:200]})
    finally:
        (folder/'harvest_audit.json').write_text(json.dumps({'target_papers':args.papers,'downloaded':len(completed),
            'discovered_unseen':len(discovered),'domains':dict(Counter(r['domain'] for r in completed.values())),
            'requests':client.log,'rejected':rejected,'rules':'fixed pre2020 CC BY/CC0 version, official URLs, cache first single connection >=4s, stop403/429, no TeX execution'},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()

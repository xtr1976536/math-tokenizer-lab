"""Small, cached, single-connection official arXiv source harvest.

Only CC BY / CC0 versions dated before 2020 enter the published formula corpus.
No HTML scraping for discovery, no bypass of rate limits, no TeX execution.
"""
import argparse,hashlib,json,time,re,xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
NS='{http://arxiv.org/OAI/arXivRaw/}'
ALLOW=('https://creativecommons.org/licenses/by/','https://creativecommons.org/publicdomain/zero/')
DOMAIN={'math.AP':'pde','math.PR':'probability','math.ST':'statistics','stat.TH':'statistics','math.NT':'number_theory',
 'math.CO':'combinatorics','math.AG':'geometry_topology','math.DG':'geometry_topology','math.GT':'geometry_topology','math.AT':'geometry_topology',
 'math.GR':'algebra','math.RA':'algebra','math.RT':'algebra','math.AC':'algebra','math.LA':'linear_algebra','math.LO':'logic',
 'math.OC':'optimization','math.NA':'analysis','math.CA':'analysis','math.CV':'analysis','math.FA':'analysis','math.DS':'ode','math.MP':'mathematical_physics'}

class OfficialClient:
    def __init__(self):
        self.session=requests.Session();self.session.headers['User-Agent']='MathTokenizerLab/0.2 noncommercial academic research; cache + single connection + 4s interval'
        self.last=0.;self.log=[]
    def get(self,url,path):
        if path.exists():return path.read_bytes()
        for attempt in range(3):
            delay=max(0,4-(time.monotonic()-self.last))
            if delay:time.sleep(delay)
            started=time.monotonic()
            try:
                response=self.session.get(url,timeout=45)
                self.last=time.monotonic()
                self.log.append({'url':url,'status':response.status_code,'duration':self.last-started,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
                # Stop rather than retry aggressively on throttling/access denial.
                if response.status_code in (403,429):raise RuntimeError('arXiv access/rate limit; harvest stopped')
                response.raise_for_status();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(response.content);return response.content
            except (requests.Timeout,requests.ConnectionError,requests.HTTPError):
                self.last=time.monotonic()
                if attempt==2:raise
                time.sleep(4*(attempt+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--papers',type=int,default=64);a=p.parse_args()
    client=OfficialClient();metadata_dir=ROOT/'data/raw/arxiv_metadata';metadata_dir.mkdir(parents=True,exist_ok=True)
    candidates={}
    # OAI dates are metadata update dates, NOT publication dates. Filter version
    # dates explicitly after discovery. Use a few small bounded record windows.
    for start,end in [('2019-01-01','2019-01-15'),('2019-06-01','2019-06-15'),('2018-06-01','2018-06-15'),('2017-06-01','2017-06-15'),('2016-06-01','2016-06-15'),('2019-12-01','2019-12-15'),('2018-01-01','2018-01-15'),('2017-01-01','2017-01-15')]:
        url=f'https://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXivRaw&set=math&from={start}&until={end}'
        path=metadata_dir/f'math-{start}-{end}.xml'
        if start=='2019-01-01' and (metadata_dir/'list.bin').exists() and not path.exists():path.write_bytes((metadata_dir/'list.bin').read_bytes())
        tree=ET.fromstring(client.get(url,path))
        for e in tree.findall('.//'+NS+'arXivRaw'):
            license=(e.findtext(NS+'license') or '').replace('http://','https://')
            if not license.startswith(ALLOW):continue
            versions=e.findall(NS+'version')
            if not versions:continue
            latest=versions[-1];date=parsedate_to_datetime(latest.findtext(NS+'date'))
            # Use latest version only if also pre-2020, so latest OAI licence is
            # consistent with downloaded article version; verify abs-page too.
            if date.year>=2020:continue
            categories=(e.findtext(NS+'categories') or '').split()
            category=next((c for c in categories if c in DOMAIN),None)
            if not category:continue
            if latest.findtext(NS+'source_type') in ('D','I'):continue
            id=e.findtext(NS+'id');version=latest.attrib['version']
            candidates[id]={'id':id,'version':version,'title':e.findtext(NS+'title'),'authors':e.findtext(NS+'authors'),
                'categories':categories,'domain':DOMAIN[category],'license_url':license,'version_date':date.isoformat(),
                'origin':'human','origin_basis':'pre2020_arxiv_historical_inference_user_approved',
                'metadata_snapshot':str(path.relative_to(ROOT)),
                'evidence':'Named academic authors; pinned article version before 2020. Historical human-origin classification requested by user; not per-author attestation.'}
        print('discovered',start,len(candidates),flush=True)
        if len(candidates)>=a.papers*2:break
    # Round-robin disciplines, deterministic ordering, no tokenizer-dependent selection.
    by={}
    for r in sorted(candidates.values(),key=lambda r:r['id']):by.setdefault(r['domain'],[]).append(r)
    selected=[]
    while any(by.values()) and len(selected)<a.papers:
        for domain in sorted(by):
            if by[domain] and len(selected)<a.papers:selected.append(by[domain].pop(0))
    registry_path=ROOT/'data/arxiv_sources.json';previous=json.loads(registry_path.read_text()) if registry_path.exists() else []
    completed={r['id']:r for r in previous};rejected=[]
    try:
        for r in selected:
            if r['id'] in completed:continue
            key=r['id'].replace('/','_')+r['version']
            url=f'https://arxiv.org/abs/{r["id"]}{r["version"]}'
            try:
                page=client.get(url,metadata_dir/(key+'.html')).decode()
                licenses=re.findall(r'https?://creativecommons\.org/(?:licenses|publicdomain)/[^"<>\s]+',page)
                matches=[x.rstrip('/')+'/' for x in licenses if (x.replace('http://','https://').rstrip('/')+'/')==r['license_url'].rstrip('/')+'/']
                if not matches:
                    rejected.append(dict(r,reason='version_license_not_verified'));continue
                srcurl=f'https://arxiv.org/src/{r["id"]}{r["version"]}'
                local='data/raw/arxiv/'+key+'.source';dest=ROOT/local
                data=client.get(srcurl,dest)
                if data.startswith(b'<!DOCTYPE') or data.startswith(b'<html') or data.startswith(b'%PDF'):
                    rejected.append(dict(r,reason='not_tex_source'));continue
                r.update({'source_url':srcurl,'abstract_url':url,'local':local,'sha256':hashlib.sha256(data).hexdigest(),
                          'license_verified_on_version_page':True,'license':'CC0-1.0' if '/zero/' in r['license_url'] else 'CC-BY-'+r['license_url'].rstrip('/').split('/')[-1]})
                completed[r['id']]=r
                registry_path.write_text(json.dumps(sorted(completed.values(),key=lambda r:r['id']),ensure_ascii=False,indent=2))
                print('downloaded',key,r['domain'],len(data),'accepted',len(completed),flush=True)
            except requests.HTTPError as exc:
                rejected.append(dict(r,reason='source_unavailable:'+str(exc)))
    finally:
        (ROOT/'data/arxiv_harvest_audit.json').write_text(json.dumps({'rules':'official OAI metadata discovery; minimum4s; one session; latest pre2020 version; version-page CC BY/CC0 verified; no nonexclusive/NC/ND redistribution',
             'discovered_permissive_candidates':len(candidates),'requested_papers':a.papers,'downloaded':len(completed),'rejected':rejected,'requests':client.log},ensure_ascii=False,indent=2))
    print('Finished',len(completed),'papers')
if __name__=='__main__':main()

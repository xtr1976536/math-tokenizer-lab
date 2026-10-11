"""New papers for a high-quality release; single cached official connection."""
import argparse
from collections import Counter, defaultdict
from email.utils import parsedate_to_datetime
import hashlib
import json
import re
import xml.etree.ElementTree as ET

from harvest_arxiv import ALLOW, DOMAIN, NS, OfficialClient, ROOT

WINDOWS = [(f'{year}-{month}-01', f'{year}-{month}-10')
           for year in (2019, 2018, 2017, 2016, 2015, 2014)
           for month in ('04', '11')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--papers', type=int, default=60)
    args = parser.parse_args()
    client = OfficialClient()
    release = ROOT / 'data/releases/v6'
    release.mkdir(parents=True, exist_ok=True)
    metadata = ROOT / 'data/raw/arxiv_metadata_v6'; metadata.mkdir(exist_ok=True)
    old = {r['id'] for path in (ROOT / 'data/arxiv_sources.json', ROOT / 'data/releases/v5/arxiv_sources.json')
           for r in json.loads(path.read_text())}
    registry = release / 'arxiv_sources.json'
    completed = {r['id']: r for r in json.loads(registry.read_text())} if registry.exists() else {}
    candidates = {}
    for start, end in WINDOWS:
        url = f'https://export.arxiv.org/oai2?verb=ListRecords&metadataPrefix=arXivRaw&set=math&from={start}&until={end}'
        location = metadata / f'math-{start}-{end}.xml'
        tree = ET.fromstring(client.get(url, location))
        for item in tree.findall('.//' + NS + 'arXivRaw'):
            identifier = item.findtext(NS + 'id')
            license_url = (item.findtext(NS + 'license') or '').replace('http://', 'https://')
            if identifier in old or not license_url.startswith(ALLOW):continue
            versions = item.findall(NS + 'version')
            if not versions:continue
            latest = versions[-1]
            date = parsedate_to_datetime(latest.findtext(NS + 'date'))
            if date.year >= 2020 or latest.findtext(NS + 'source_type') in ('D', 'I'):continue
            categories = (item.findtext(NS + 'categories') or '').split()
            primary = next((category for category in categories if category in DOMAIN), None)
            if primary is None:continue
            candidates[identifier] = {'id': identifier, 'version': latest.attrib['version'],
                'version_date': date.isoformat(), 'title': item.findtext(NS + 'title'),
                'authors': item.findtext(NS + 'authors'), 'categories': categories, 'primary_category': primary,
                'domain': DOMAIN[primary], 'license_url': license_url,
                'metadata_snapshot': str(location.relative_to(ROOT)),
                'origin': 'human', 'origin_basis': 'pre2020_arxiv_historical_inference_user_approved',
                'evidence': 'Named academic authors, fixed source version before2020; historical inference, not per-author attestation.'}
        counts = Counter(r['domain'] for r in candidates.values())
        print('new discovery', start, len(candidates), dict(counts), flush=True)
        targets = ['pde', 'algebra', 'geometry_topology', 'number_theory', 'combinatorics', 'logic']
        if len(candidates) >= args.papers * 2 and all(counts[d] >= 5 for d in targets):break
    buckets = defaultdict(list)
    for record in sorted(candidates.values(), key=lambda r: r['id']):buckets[record['domain']].append(record)
    order = []
    while any(buckets.values()):
        for domain in sorted(buckets):
            if buckets[domain]:order.append(buckets[domain].pop(0))
    rejected = []
    try:
        for record in order:
            if len(completed) >= args.papers:break
            if record['id'] in completed:continue
            key = record['id'].replace('/', '_') + record['version']
            try:
                abstract = f'https://arxiv.org/abs/{record["id"]}{record["version"]}'
                page = client.get(abstract, metadata / f'{key}.html').decode()
                links = re.findall(r'https?://creativecommons\.org/(?:licenses|publicdomain)/[^"<>\s]+', page)
                if not any(u.replace('http://', 'https://').rstrip('/') == record['license_url'].rstrip('/') for u in links):
                    rejected.append(dict(record, reason='fixed_version_license_mismatch'));continue
                url = f'https://arxiv.org/src/{record["id"]}{record["version"]}'
                local = f'data/raw/arxiv_v6/{key}.source'
                raw = client.get(url, ROOT / local)
                if raw.startswith((b'%PDF', b'<!DOCTYPE', b'<html')):
                    rejected.append(dict(record, reason='not_TeX_source'));continue
                record.update(source_url=url, abstract_url=abstract, local=local,
                              sha256=hashlib.sha256(raw).hexdigest(), license_verified_on_version_page=True,
                              license='CC0-1.0' if '/zero/' in record['license_url'] else 'CC-BY-' + record['license_url'].rstrip('/').split('/')[-1])
                completed[record['id']] = record
                registry.write_text(json.dumps(sorted(completed.values(), key=lambda r: r['id']), ensure_ascii=False, indent=2) + '\n')
                print('new source', key, record['domain'], len(completed), flush=True)
            except RuntimeError:raise
            except Exception as exc:rejected.append(dict(record, reason='unavailable:' + str(exc)[:160]))
    finally:
        audit = {'requested': args.papers, 'downloaded': len(completed), 'candidates': len(candidates),
                 'requests': client.log, 'rejected': rejected,
                 'rules': 'official OAI/source URLs; cached single connection >=4s; stop on403/429; fixed CC BY/CC0 pre2020; source never executed'}
        (release / 'harvest_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print('harvest complete', len(completed), flush=True)


if __name__ == '__main__':main()

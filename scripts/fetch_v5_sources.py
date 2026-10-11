import json,hashlib
from pathlib import Path
from harvest_arxiv import OfficialClient,ROOT
client=OfficialClient()
for source in json.loads((ROOT/'data/releases/v5/arxiv_sources.json').read_text()):
 data=client.get(source['source_url'],ROOT/source['local']);assert hashlib.sha256(data).hexdigest()==source['sha256'];print('verified',source['id']+source['version'],flush=True)
print('v5 pinned source snapshot replay verified')

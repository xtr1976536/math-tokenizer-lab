"""Lossless gzip master corpus storage for GitHub's individual-file limit."""
import gzip,hashlib,shutil,json
from .corpus import ROOT

def ensure_master():
 folder=ROOT/'data/releases/v5/curated';path=folder/'corpus.jsonl';archive=folder/'corpus.jsonl.gz'
 if not path.exists() and archive.exists():
  with gzip.open(archive,'rb') as source,path.open('wb') as target:shutil.copyfileobj(source,target)
 if path.exists() and (folder/'summary.json').exists():
  expected=json.loads((folder/'summary.json').read_text())['corpus_sha256']
  actual=hashlib.sha256(path.read_bytes()).hexdigest()
  if actual!=expected:raise RuntimeError('master corpus integrity mismatch')
 return path

def archive_master():
 path=ensure_master();archive=path.with_suffix(path.suffix+'.gz')
 with path.open('rb') as source,archive.open('wb') as stream:
  with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0) as target:shutil.copyfileobj(source,target)
 with gzip.open(archive,'rb') as stream:assert hashlib.sha256(stream.read()).hexdigest()==hashlib.sha256(path.read_bytes()).hexdigest()
 return archive

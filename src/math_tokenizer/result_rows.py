"""Bounded-memory sequential result archive reader for large full replays."""
import gzip,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory


class PerFormulaStore:
    def __init__(self,archive):
        self.temporary=TemporaryDirectory(prefix='math-tokenizer-rows-')
        self.root=Path(self.temporary.name);self.paths={};self.last=None;self.cached=None
        current=None;stream=None;closed=set()
        try:
            with gzip.open(archive,'rt') as source:
                for line in source:
                    row=json.loads(line);key=(row['key'],row['suite'])
                    if key!=current:
                        if stream is not None:stream.close();closed.add(current)
                        if key in closed:raise ValueError('Result key is noncontiguous; unsafe sequential spool')
                        path=self.root/(hashlib.sha256(repr(key).encode()).hexdigest()+'.jsonl.gz')
                        self.paths[key]=path;stream=gzip.open(path,'wt');current=key
                    stream.write(line)
        finally:
            if stream is not None:stream.close()
    def __getitem__(self,key):
        if key!=self.last:
            with gzip.open(self.paths[key],'rt') as stream:self.cached=list(map(json.loads,stream))
            self.last=key
        return self.cached
    def close(self):self.cached=None;self.temporary.cleanup()

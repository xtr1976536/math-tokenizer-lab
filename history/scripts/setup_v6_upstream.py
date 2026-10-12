"""Pinned original metric and paper implementations, no upstream edits."""
import argparse,os,subprocess,json
from pathlib import Path
from math_tokenizer.corpus import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--rust-bin',default=str(ROOT/'runtime/cargo/bin'));args=p.parse_args()
    registry=json.loads((ROOT/'references/upstream_code.json').read_text())
    for name in ('tokeval','pathpiece','fungrim'):
        r=registry[name];path=ROOT/r['local_clone']
        if not path.exists():subprocess.run(['git','clone',r['repo'],str(path)],check=True)
        if subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()!=r['revision']:
            assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain'],text=True).strip(),'preserve local upstream edits'
            subprocess.run(['git','-C',str(path),'checkout',r['revision']],check=True)
    subprocess.run(['uv','pip','install','--python',str(ROOT/'.venv/bin/python'),'-e',str(ROOT/registry['tokeval']['local_clone'])],check=True)
    environment=os.environ.copy();environment['PATH']=args.rust_bin+os.pathsep+environment['PATH']
    if (ROOT/'runtime/rustup').exists():environment['RUSTUP_HOME']=str(ROOT/'runtime/rustup');environment['CARGO_HOME']=str(ROOT/'runtime/cargo')
    wheels=ROOT/'runtime/wheels';wheels.mkdir(parents=True,exist_ok=True)
    subprocess.run([str(ROOT/'.venv/bin/maturin'),'build','--release','--manifest-path',str(ROOT/'vendor/pathpiece/pathpiece/Cargo.toml'),'--interpreter',str(ROOT/'.venv/bin/python'),'--out',str(wheels)],env=environment,check=True)
    wheel=max(wheels.glob('pathpiece-*.whl'),key=lambda p:p.stat().st_mtime)
    subprocess.run(['uv','pip','install','--python',str(ROOT/'.venv/bin/python'),str(wheel)],check=True)

if __name__=='__main__':main()

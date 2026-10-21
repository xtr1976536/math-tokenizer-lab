"""Seal reproducible artifacts only after all requested experiments pass audits."""
import hashlib,importlib.metadata,json,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'experiments/zip_ast_20261021'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 completion=json.loads((O/'completion.json').read_text());assert completion['status']=='complete' and completion['audits']==['PASS']*3
 audits=[json.loads((O/f'audit_round_{i}.json').read_text()) for i in (1,2,3)]
 assert all(a['scope']=='complete' for a in audits)
 assert audits[1]['hard_native_python_reference_cases']==768 and audits[1]['soft_native_independent_dynamic_program_cases']==288
 assert audits[2]['result_files_checked']==228
 protocol=json.loads((O/'protocol.json').read_text());assert len(protocol['splits'])==12
 scripts={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'scripts/zip_ast_20261021').glob('*')) if p.is_file()}
 results={str(p.relative_to(O)):sha(p) for sp in protocol['splits'] for p in sorted((O/sp['name']).rglob('*')) if p.is_file()}
 metadata=dict(status='SEALED',catalog_sha256=sha(ROOT/'data/benchmark/metadata/formulas.jsonl.gz'),protocol_sha256=sha(O/'protocol.json'),initial_runner_snapshot_sha256=sha(O/'protocol_runner_snapshot.py.txt'),scripts=scripts,results=results,report_sha256=sha(O/'REPORT_ZH.md'),baseline_provenance_sha256=sha(O/'baseline_provenance.json'),zip_provenance_sha256=sha(O/'zip_provenance.json'),python=platform.python_version(),packages={k:importlib.metadata.version(k) for k in ['numpy','tokenizers','tiktoken','regex','sentencepiece','matplotlib']},git_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),split_seed_role='Only source partition randomization; unmodified ZIP default training settings; no pretrained research vocab loaded',note='Runner portability fallback and PDE split-diversity audit were finalized before corresponding fits. Model algorithms and four metric definitions unchanged. Initial-overlap PDE diagnostics kept outside final experiment and excluded from all reported results.')
 (O/'execution_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
 print('SEALED',len(results),'result/checkpoint files; dataset and sources hashed')
if __name__=='__main__':main()

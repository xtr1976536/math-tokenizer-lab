import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from math_tokenizer.corpus import ROOT
from math_tokenizer.evaluate import file_hash,read_jsonl


def test_v4_corpus_and_no_baseline_locked():
 for cap in (512,1024):
  path=ROOT/f'results/v4_{cap}/protocol.json'
  if not path.exists():continue
  p=json.loads(path.read_text());assert p['corpus_sha256']==file_hash(ROOT/'data/releases/v2/curated/corpus.jsonl')
  assert p['new_baselines_run'] is False and p['capacity']==cap
  assert set(p['train_ids']).isdisjoint(p['dev_ids']);assert set(p['train_ids']).isdisjoint(p['test_ids'])


def test_selected_policy_contains_actual_updates_and_no_teacher():
 path=ROOT/'results/v4_512/models/rl_v4-17.json'
 if not path.exists():
  from math_tokenizer.checkpoints import retirement_entry
  entry=retirement_entry(path)
  assert entry is not None and entry['relocated_to'] is None
  # The old trained RL-v4 policy is intentionally retired. Its independent
  # learning audit stays as historical evidence; it is not a current model.
  evidence=json.loads((ROOT/'results/retrospective/evidence.json').read_text())['v4_learning_vs_initial_profile']['512']['profile']
  assert evidence['trained']['mean_expected_tokens_on_identical_training_subset']<evidence['matched_initial']['mean_expected_tokens_on_identical_training_subset']
  return
 d=json.loads(path.read_text());assert d['format']=='masked-rl-math-v4'
 log=d['training_log'][-1];assert log['teacher_labels'] is False and log['shortest_path_targets'] is False
 assert len(log['epochs'])>=12
 assert any(abs(x)>1e-6 for x in d['theta'])
 assert log['epochs'][-1]['train_expected_tokens']<log['epochs'][0]['train_expected_tokens']
 assert all(r['scope']=='train_only' for r in log['epochs'])


def test_command_coverage_not_claimed_universal():
 path=ROOT/'results/v4_512/metrics.json'
 if not path.exists():return
 m=json.loads(path.read_text());rr=[r for r in m if r['method']=='rl_v4']
 for r in rr:
  assert r['hybrid_tree_crossing_rate']==0 and r['roundtrip_rate']==1
  assert 'atomic_escape_counts' in r and r['atomic_escape_formulas']>0

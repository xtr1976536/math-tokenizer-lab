"""Final v5 choice uses dev only; an empty barrier is not an RL improvement."""
import json,time
import numpy as np
from math_tokenizer.corpus import ROOT
from train_routes_v6 import write

for scenario in ('broad','specialized'):
    out=ROOT/f'results/v6/{scenario}_1024'
    trajectory=json.loads((out/'rl_selection.json').read_text());structural=json.loads((out/'structural_selection.json').read_text());barrier=json.loads((out/'barrier_selection.json').read_text())
    trajectory_score=trajectory['mean_dev_utility'][trajectory['selected']];structural_score=structural['selected']['score']['utility']
    # User permits a small quality trade for a large speed improvement. Exact
    # bandit training runs one merge trajectory rather than eight complete
    # trajectories for terminal RLOO. This margin is frozen before test scores.
    selected='structural' if structural_score<=trajectory_score+.002 else 'trajectory'
    paths=[out/'models'/f'v5_selected_{seed}.json' for seed in (17,29,43)]+[out/'models/v5_structural_selected_17.json']
    assert all(p.exists() for p in paths)
    result={'v5_branch':selected,'dev_utility':{'trajectory':trajectory_score,'structural':structural_score,'barrier':barrier['mean_dev_utility']},
      'selection_rule':'prefer deterministic structural RL when utility within .002 of trajectory RL; otherwise lower utility; pre-test compute/quality trade',
      'accepted_dev_utility_loss':max(0,structural_score-trajectory_score) if selected=='structural' else 0,
      'barrier_is_research_only':'All selected masks empty; no improvement attributable to RL over its same-atom frequency control.',
      'selected_v5_paths':[str(p.relative_to(ROOT)) for p in ([out/'models/v5_structural_selected_17.json'] if selected=='structural' else paths[:3])],
      'test_used':False,'frozen_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'ast':'models/ast_v4_selected.json'}
    write(out/'final_selection.json',result);print(scenario,result,flush=True)

"""Load-only numerical cache with exact integrity and bounded device replay.

No fit function is imported or called. Different CPU/MPS kernels can produce
non-bit-identical predictions; weights, coordinates and stored predictions
remain byte-exact, while new predictions are checked with fixed tolerances.
"""
import hashlib,json
from pathlib import Path
import torch
from .author import tensor_hash
from .data import case_environment,build_surrogate
from .expanded_corpus import sha


def load_existing(folder,case,device='cpu',seed=525,observations=10000):
 folder=Path(folder);cache=folder/'surrogate_cache.pt'
 if not cache.exists():raise FileNotFoundError('Existing fitted MLP required; automatic refit forbidden: '+str(cache))
 before=sha(cache);quality=json.loads((folder/'surrogate_quality.json').read_text())
 assert before==quality['cache_sha256']
 saved=torch.load(cache,map_location='cpu',weights_only=True);report=dict(saved['report'])
 assert (report['equation'],report['seed'],report['noise_percent'],report['observations'])==(case,seed,0,observations)
 assert tensor_hash(saved['state'])==report['surrogate_sha256']
 assert hashlib.sha256(saved['database'].numpy().tobytes()).hexdigest()==report['metadata_sha256']
 assert hashlib.sha256(saved['predictions'].numpy().tobytes()).hexdigest()==report['base_predictions_sha256']
 env=case_environment(case,observations,0,device,seed)
 assert hashlib.sha256(env['un'].tobytes()).hexdigest()==report['raw_noisy_data_sha256']
 net=build_surrogate(env,device,seed);net.load_state_dict(saved['state']);net.eval()
 database=saved['database'].to(device).requires_grad_(True)
 with torch.no_grad():predictions=net(database).cpu()
 expected=saved['predictions'];delta=(predictions-expected).abs()
 assert torch.isfinite(predictions).all() and torch.allclose(predictions,expected,rtol=1e-5,atol=1e-5),float(delta.max())
 report['cache_validation']=dict(cache_sha256=before,device=device,exact_weight_and_coordinate_hashes=True,
      stored_prediction_hash_exact=True,prediction_replay_rtol=1e-5,prediction_replay_atol=1e-5,
      maximum_absolute_difference=float(delta.max()),fit_called=False)
 assert sha(cache)==before
 return env,net,database,saved['state'],report

"""Dev-only native dictionary refinement, still before first v6 test scoring."""
import argparse,json,time
from math_tokenizer.corpus import ROOT
from math_tokenizer.benchmark_v6 import master_path
from math_tokenizer.ast_route_v6 import NativeASTV4
from math_tokenizer.dictionary_v6 import augment_native
from train_routes_v6 import score,write

p=argparse.ArgumentParser();p.add_argument('--scenario',default='broad');p.add_argument('--capacity',type=int,default=1024);a=p.parse_args()
out=ROOT/f'results/v6/{a.scenario}_{a.capacity}';scope=json.loads((out/'scope.json').read_text());wanted=set(scope['fit_ids']+scope['reward_ids']+scope['dev_ids']);byid={r['id']:r for r in map(json.loads,master_path().open()) if r['id'] in wanted}
fit=[byid[id] for id in scope['fit_ids']];reward=[byid[id] for id in scope['reward_ids']];dev=[byid[id] for id in scope['dev_ids']]
if (out/'models/ast_v4_native_before_dictionary.json').exists():base=NativeASTV4.load(out/'models/ast_v4_native_before_dictionary.json')
else:
    base=NativeASTV4.load(out/'models/ast_v4_selected.json');base.save(out/'models/ast_v4_native_before_dictionary.json')
stats=[];models=[]
for extra in (256,768):
    model=augment_native(base,fit,reward,a.capacity,extra=extra);stats.append({'extra':extra,'score':score(model,dev,'ast_dictionary_'+str(extra)),'train_seconds':model.train_seconds});models.append(model)
    model.save(out/f'models/ast_v4_dictionary_{extra}.json');print(a.scenario,a.capacity,stats[-1],flush=True)
before=score(base,dev,'native_before_dictionary');winner=min(range(len(stats)),key=lambda i:stats[i]['score']['utility'])
selected=models[winner] if stats[winner]['score']['utility']<before['utility'] else base;selected.save(out/'models/ast_v4_selected.json')
write(out/'ast_dictionary_dev.json',{'before':before,'variants':stats,'promoted':selected is not base,'selected_extra':stats[winner]['extra'] if selected is not base else 0,'test_used':False})
selection=json.loads((out/'ast_selection.json').read_text());selection['dictionary_refinement']=json.loads((out/'ast_dictionary_dev.json').read_text());selection['frozen_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());write(out/'ast_selection.json',selection)

"""Fresh-process load construction and peak RSS, corpus allocation excluded."""
import argparse,json,subprocess,sys,time,resource,platform
from math_tokenizer.corpus import ROOT
from math_tokenizer.experiment_v6 import load_entry

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);args=p.parse_args();folder=ROOT/'results/v6/full';entries=json.loads((folder/'registry.json').read_text())
    if args.worker is not None:
        entry=entries[args.worker];before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;start=time.perf_counter();model=load_entry(entry);elapsed=time.perf_counter()-start
        model.encode(r'\frac{x^2}{y}+\alpha=12345');after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor=1024**2 if platform.system()=='Darwin' else 1024
        print(json.dumps({'key':entry['key'],'load_seconds':elapsed,'baseline_process_peak_mib':before/divisor,'loaded_process_peak_mib':after/divisor,'incremental_peak_mib':max(0,after-before)/divisor,'scope':'fresh process, common imports, model construction plus one fixed probe, no corpus'}));return
    output=[]
    for i,entry in enumerate(entries):
        # Load/resource reports are engineering descriptors, not ranking weights.
        process=subprocess.run([sys.executable,str(ROOT/'scripts/profile_deployment_v6.py'),'--worker',str(i)],capture_output=True,text=True,check=True)
        output.append(json.loads(process.stdout.splitlines()[-1]))
    (folder/'deployment_profile.json').write_text(json.dumps(output,indent=2)+'\n');print('deployment profiles',len(output),flush=True)

if __name__=='__main__':main()

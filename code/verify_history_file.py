"""Independent file/trace/policy/quality alignment checks for the completed study."""
from pathlib import Path
import argparse,csv,hashlib,json,math
import numpy as np
import exp_file_controlled as base
import file_workload as workload
from anq.file_store import RECORD_DTYPE

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--reference-populations',default='data/population_manifest.json')
    a=ap.parse_args();study=ROOT/a.study;out=ROOT/a.out
    meta=json.loads((study/'manifest.json').read_text(encoding='utf8'))
    assert meta['status']=='COMPLETE' and not (study/'INCOMPLETE').exists()
    sessions=list(csv.DictReader((study/'sessions.csv').open(encoding='utf8')))
    queries=list(csv.DictReader((study/'queries.csv').open(encoding='utf8')))
    source={}; checked_pairs=checked_queries=prefix_cases=0;max_prefix_ulps=0.
    for session in sessions:
        path=study/session['source_file']
        if path not in source:source[path]=np.memmap(path,mode='r',dtype=RECORD_DTYPE)
        physical=source[path];specs=json.loads(session['specs_json'])
        own=[x for x in queries if x['session_id']==session['session_id']]
        with np.load(study/session['trace_file']) as z:
            for row in own:
                k=int(row['query']);pos=z[f'q{k}_positions'];u=z[f'q{k}_u'];v=z[f'q{k}_v']
                assert len(pos)==len(u)==len(v)==int(row['n'])
                assert len(np.unique(pos))==len(pos)
                eu,ev=workload.evaluate(physical[pos],specs[k],validate=False)
                assert np.array_equal(eu,u) and np.array_equal(ev,v),(session['session_id'],k)
                assert base.array_sha(pos)==row['draw_positions_sha256']
                assert hashlib.sha256(u.tobytes()+v.tobytes()).hexdigest()==row['pairs_sha256']
                checked_pairs+=len(pos);checked_queries+=1
                if (session['method']=='rebase' and int(session['replicate'])==0 and
                        k in (0,int(session['queries'])-1)):
                    n=int(row['prefix_rows']);pu,pv=workload.evaluate(physical[:n],specs[k],validate=False)
                    stored=json.loads(row['prefix_moments_json']);assert stored['n']==n
                    for name,values in [('su',pu),('sv',pv)]:
                        reference=math.fsum(values);ulp=math.ulp(reference)
                        error=abs(stored[name]-reference)/ulp
                        max_prefix_ulps=max(max_prefix_ulps,error)
                        assert error<=2,(session['session_id'],k,name,error)
                    prefix_cases+=1
    # Check all physical sources against the original recorded byte hashes.
    prep=list(csv.DictReader((study/'source_preparation.csv').open(encoding='utf8')))
    for row in prep:
        path=study/'physical_sources'/f"{row['dataset']}_{row['replicate']}.bin"
        assert base.sha(path)==row['source_sha256'],path
    original=json.loads((ROOT/a.reference_populations).read_text())
    for dataset,input_meta in meta['input'].items():
        assert input_meta['normalized_arrays_sha256']==original['input'][dataset]['normalized_arrays_sha256']
    result=dict(status='PASS',sessions=len(sessions),queries=checked_queries,retained_sample_pairs=checked_pairs,
        all_saved_pairs_match_selected_functions=True,all_wor_positions_unique=True,
        source_files_hash_verified=len(prep),same_normalized_populations_as_existing_paper=True,
        independently_summed_rebase_prefix_cases=prefix_cases,max_prefix_first_moment_ulps=max_prefix_ulps,
        selection_replay_audit=meta['audit'])
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='selection_replay_audit'},indent=2))


if __name__=='__main__':main()

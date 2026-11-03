"""Rebuild omitted physical source files from bundled exact inputs and recorded seeds."""
from pathlib import Path
import argparse,csv,json
import exp_file_controlled as base

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    study=ROOT/a.study;meta=json.loads((study/'manifest.json').read_text(encoding='utf8'))
    rows=list(csv.DictReader((study/'source_preparation.csv').open(encoding='utf8')))
    base.original_loader.DATA_ROOT=ROOT/'data';rebuilt=[]
    (study/'physical_sources').mkdir(exist_ok=True)
    for dataset in meta['input']:
        data,loaded=base.original_loader.load(dataset)
        assert loaded['normalized_arrays_sha256']==meta['input'][dataset]['normalized_arrays_sha256']
        for row in rows:
            if row['dataset']!=dataset:continue
            target=study/'physical_sources'/f"{dataset}_{row['replicate']}.bin"
            if target.exists():
                if base.sha(target)!=row['source_sha256']:raise FileExistsError('Conflicting source: '+str(target))
                continue
            generated=base.physical_file(target,data,int(row['source_seed']))
            assert generated['source_sha256']==row['source_sha256']
            assert generated['order_sha256']==row['order_sha256']
            rebuilt.append(target.name);print('REBUILT',target.name,flush=True)
        del data
    print('Verified/rebuilt',len(rows),'physical sources;',len(rebuilt),'new files')


if __name__=='__main__':main()

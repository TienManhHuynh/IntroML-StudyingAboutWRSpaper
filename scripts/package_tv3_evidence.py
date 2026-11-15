"""Copy small measured development evidence into versioned results, excluding raw data."""
import csv
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    """Copy completed local run evidence, pin verified source checksum, and update coverage."""
    out=ROOT/'results/development/sst2-baseline'
    source=ROOT/'results/raw/tv3-hosted'
    copied=0
    for run in sorted(source.glob('*/*')):
        state=json.loads((run/'run.json').read_text())
        if state['status']!='completed_local' or not state['development_run']:
            raise ValueError('Expected completed local development evidence')
        target=out/'evidence'/run.parent.name/run.name
        target.mkdir(parents=True,exist_ok=True)
        for name in ('config.json','environment.json','checkpoints.csv','metrics.json','run.json','split-indices.npz'):
            shutil.copyfile(run/name,target/name)
        copied+=1
    if copied!=10:
        raise ValueError('Expected 10 baseline runs (2 learners x 5 seeds)')
    metadata=json.loads((source/'pac/42/config.json').read_text(encoding='utf-8'))['data']
    spec_path=ROOT/'configs/data_sources.json'
    specs=json.loads(spec_path.read_text())
    specs['sst2']['sha256']=metadata['sha256']
    spec_path.write_text(json.dumps(specs,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    registry_path=ROOT/'configs/datasets.json'
    registry=json.loads(registry_path.read_text())
    for item in registry['datasets']:
        if item['id']=='sst2':
            item.update(hosted_source='https://huggingface.co/datasets/stanfordnlp/sst2',hosted_revision=metadata['revision'],
                        download_status='hf_asset_verified',preprocessing_status='development_only_equivalence_pending')
    registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    coverage_path=ROOT/'results/coverage.csv'
    with coverage_path.open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    for row in rows:
        if row['group']=='MAIN' and row['dataset']=='sst2':
            row.update(status='development_only',artifact='results/development/sst2-baseline/summary.json',note='10 local baseline trials on a 600-sample development subset; WRS/full reproduction/public W&B pending')
    with coverage_path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print('Packaged 10 development runs without raw features; no reproduction completion claimed.')


if __name__=='__main__':
    main()

"""Summarize compatible measured runs and create traceable plots, without invented values."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.run_reproduction import write_json


def summarize(paths, output):
    """Write run-index/summary/plots for completed evidence directories and return counts.

    Grouping includes dataset revision, subset, preprocessing, model and all algorithm
    parameters. Duplicate seed/config runs are rejected; incomplete trials are disclosed.
    """
    groups, index = defaultdict(list), []
    for path in map(Path, paths):
        state = json.loads((path/'run.json').read_text())
        if state['status'] not in ('completed_local','completed_online'):
            raise ValueError(f'Refusing incomplete/failed run: {path}')
        config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        metrics = json.loads((path/'metrics.json').read_text())
        environment = json.loads((path/'environment.json').read_text())
        data = dict(config['data'])
        # Split/vocabulary checksum changes by seed; retain preprocessing recipe, source and sizes.
        for key in ('split_seed','split_sha256','vocabulary_sha256','subset_indices_sha256','n_features'):
            data.pop(key, None)
        signature = {key:config.get(key) for key in ('learner','C_err','eta','lambda_','K','weighting','averaging','zeroing','noise_rate','include_wrs')}
        signature.update(data=data, git_commit=environment['git_commit'], dirty=environment['dirty'])
        for model, values in metrics.items():
            ident = json.dumps(dict(signature,model=model),sort_keys=True)
            if any(x['seed']==config['seed'] for x in groups[ident]):
                raise ValueError('Duplicate seed within a compatible configuration group')
            row = dict(seed=config['seed'], path=str(path), wandb_url=state.get('wandb_url'), **values)
            groups[ident].append(row)
            index.append(dict(dataset=config['dataset'], learner=config['learner'], model=model, seed=config['seed'],
                              noise_rate=config.get('noise_rate',0), revision=config['data'].get('revision'),
                              wandb_url=state.get('wandb_url') or '', path=str(path), development=state['development_run']))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    for ident, rows in groups.items():
        item = {'configuration':json.loads(ident), 'n_trials':len(rows), 'five_trials_complete':len(rows)==5,
                'all_public_run_urls_present':all(row['wandb_url'] for row in rows), 'runs':rows}
        for metric in ('final_accuracy','final_sparsity','rop','train_seconds'):
            values = np.array([row[metric] for row in rows])
            item[metric] = {'mean':float(values.mean()),'min':float(values.min()),'max':float(values.max())}
        summary.append(item)
    write_json(out/'summary.json', summary)
    if index:
        with (out/'run-index.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(index[0])); writer.writeheader(); writer.writerows(index)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for i, path in enumerate(map(Path, paths)):
        with (path/'checkpoints.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        fig, axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for model in sorted({row['model'] for row in rows}):
            selected=[row for row in rows if row['model']==model]
            steps=[int(row['step']) for row in selected]
            axes[0].plot(steps,[float(row['accuracy']) for row in selected],label=model)
            axes[1].plot(steps,[float(row['sparsity']) for row in selected],label=model)
        for ax,label in zip(axes,('Test accuracy (fraction)','Model sparsity (fraction)')):
            ax.set(xlabel='Training examples processed',ylabel=label); ax.legend(); ax.grid(alpha=.25)
        fig.suptitle(str(path.name)+' | measured run; see run-index for provenance')
        fig.savefig(out/f'trajectory-{i}.png',dpi=160)
        svg=out/f'trajectory-{i}.svg'
        fig.savefig(svg)
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
        plt.close(fig)
    return {'runs':len(paths),'configuration_groups':len(groups)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); print(summarize(args.runs,args.output))

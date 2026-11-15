"""Paired single-pass experiment runner. Use --help for explicit development/full options."""
import argparse
import csv
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data.loader import load_dataset, split_data, training_label_noise
from src.evaluation.metrics import accuracy, sparsity, relative_oracle_performance, checkpoint_steps
from src.evaluation.tracking import track_run
from src.models.pac import PAC
from src.models.fsol import FSOL


def write_json(path, value):
    """Write UTF-8 strict JSON evidence to path, rejecting NaN/Infinity."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def run_experiment(config, output, bundle=None):
    """Run one paired trial, save evidence and track it; return run metadata.

    config supplies learner/seed/parameters, optional WAT factory, tracking and dataset.
    output must not exist. bundle optionally supplies validated pre-split test data.
    Returns: run metadata; failures leave status=failed and raise the original error.
    """
    factory = None
    if config.get('include_wrs'):
        if not config.get('wat_factory'):
            raise ValueError('WAT implementation pending: supply --wat-factory module:callable from TV2')
        module, name = config['wat_factory'].split(':')
        factory = getattr(importlib.import_module(module), name)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    status = {'status': 'running', 'wandb_url': None, 'reportable': False}
    write_json(out/'run.json', status)
    try:
        if bundle is None:
            if config['dataset'] == 'synthetic':
                from sklearn.datasets import make_classification
                X, y = make_classification(n_samples=config.get('samples', 300), n_features=20, n_informative=8, random_state=config['seed'])
                bundle = split_data(X, 2*y-1, seed=config['seed'], metadata={'dataset': 'synthetic', 'development_subset': True})
            else:
                bundle = load_dataset(config['dataset'], seed=config['seed'], source_file=config.get('sources', str(ROOT/'configs/data_sources.json')), max_samples=config.get('max_samples'))
        from src.data.loader import validate_features
        validate_features(bundle.X_train, bundle.y_train); validate_features(bundle.X_test, bundle.y_test)
        config = dict(config, data=bundle.metadata)
        config['noise_seed'] = config.get('noise_seed') if config.get('noise_seed') is not None else config['seed'] + 100000
        noisy, mask = training_label_noise(bundle.y_train, config.get('noise_rate', 0), config['noise_seed'])
        config['noise_flipped'] = int(mask.sum())
        write_json(out/'config.json', config)
        np.savez_compressed(out/'split-indices.npz', train=bundle.train_indices, test=bundle.test_indices, noise_mask=mask)
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        versions = {}
        for package in ('numpy', 'scipy', 'scikit-learn', 'wandb', 'huggingface-hub', 'pyarrow'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json(out/'environment.json', {'git_commit': commit, 'dirty': dirty, 'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions})
        def base():
            return PAC(bundle.X_train.shape[1], C_err=config['C_err']) if config['learner'] == 'pac' else FSOL(bundle.X_train.shape[1], eta=config['eta'], lambda_=config['lambda_'])
        models = {'baseline': base()}
        if factory:
            models['wrs'] = factory(base_learner=base(), K=config['K'], weighting=config['weighting'], averaging=config['averaging'], voting_zeroing=config['zeroing'], seed=config['seed'] + 200000)
        elapsed = {name: 0.0 for name in models}
        checkpoints, evaluation_seconds = [], 0.0
        steps = set(checkpoint_steps(len(noisy), config.get('checkpoints', 200)).tolist())
        for step in range(len(noisy)+1):
            if step in steps:
                start = time.perf_counter()
                for name, model in models.items():
                    checkpoints.append({'step': step, 'model': name, 'accuracy': accuracy(bundle.y_test, model.predict(bundle.X_test)), 'sparsity': sparsity(model.get_weights())})
                evaluation_seconds += time.perf_counter() - start
            if step < len(noisy):
                x = bundle.X_train[step]
                for name, model in models.items():
                    start = time.perf_counter()
                    model.update(x, int(noisy[step]))
                    elapsed[name] += time.perf_counter() - start
        baseline = [r['accuracy'] for r in checkpoints if r['model'] == 'baseline']
        metrics = {}
        for name in models:
            rows = [r for r in checkpoints if r['model'] == name]
            metrics[name] = {'final_accuracy': rows[-1]['accuracy'], 'final_sparsity': rows[-1]['sparsity'],
                             'rop': relative_oracle_performance(baseline, [r['accuracy'] for r in rows]),
                             'train_seconds': elapsed[name], 'evaluation_seconds_total': evaluation_seconds}
        with (out/'checkpoints.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'model', 'accuracy', 'sparsity'])
            writer.writeheader(); writer.writerows(checkpoints)
        write_json(out/'metrics.json', metrics)
        url = track_run(config, checkpoints, metrics, out, mode=config.get('wandb_mode', 'disabled'), project=config.get('wandb_project'), entity=config.get('wandb_entity'))
        development = bool(bundle.metadata.get('development_subset')) or config['dataset'] == 'synthetic'
        status.update(status='completed_online' if url else 'completed_local', wandb_url=url,
                      reportable=False, development_run=development,
                      note='Successful execution is not evidence of paper equivalence. Review dataset/protocol and public W&B before reporting.')
        write_json(out/'run.json', status)
        return status
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__)
        write_json(out/'run.json', status)
        raise


def parser():
    """Return the shared baseline/noise CLI argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='synthetic')
    p.add_argument('--learner', choices=['pac', 'fsol'], default='pac')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--C-err', dest='C_err', type=float, default=1)
    p.add_argument('--eta', type=float, default=1)
    p.add_argument('--lambda', dest='lambda_', type=float, default=.01)
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--max-samples', type=int)
    p.add_argument('--checkpoints', type=int, default=200)
    p.add_argument('--sources', default=str(ROOT/'configs/data_sources.json'))
    p.add_argument('--include-wrs', action='store_true')
    p.add_argument('--wat-factory', help='TV2 module:callable; see docs/tv3-running.md')
    p.add_argument('--K', type=int, choices=[1,4,16,64], default=16)
    p.add_argument('--weighting', choices=['standard','exponential'], default='standard')
    p.add_argument('--averaging', choices=['simple','weighted'], default='simple')
    p.add_argument('--zeroing', action='store_true')
    p.add_argument('--noise-rate', type=float, default=0)
    p.add_argument('--noise-seed', type=int)
    p.add_argument('--wandb-mode', choices=['disabled','offline','online'], default='disabled')
    p.add_argument('--wandb-project', default=os.environ.get('WANDB_PROJECT','wrs-reproduction-group8'))
    p.add_argument('--wandb-entity', default=os.environ.get('WANDB_ENTITY') or None)
    p.add_argument('--output', type=Path, required=True)
    return p


def main():
    """Run explicit CLI configuration; exit nonzero on data, model or tracking failure."""
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    args = vars(parser().parse_args())
    out = args.pop('output')
    args['experiment'] = 'reproduction-development' if args['dataset']=='synthetic' or args['max_samples'] else 'reproduction'
    print(json.dumps(run_experiment(args, out), indent=2))


if __name__ == '__main__':
    main()

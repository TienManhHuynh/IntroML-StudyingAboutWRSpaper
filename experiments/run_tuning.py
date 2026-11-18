"""Appendix B.2 base-learner grid search, with an explicit top-2.5% rounding policy."""
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_reproduction import parser, run_experiment, write_json
from src.data.loader import load_dataset, split_data


def select_parameters(candidates):
    """Choose highest mean ROP within ceil(2.5% * configuration count) accuracy leaders.

    candidates: list of {parameters, trials: [{seed,final_accuracy,rop},...]},
    requiring the same five distinct seeds for every configuration. Returns selection.
    Ties are deterministic by parameter JSON; rounding is a disclosed implementation choice.
    """
    if not candidates:
        raise ValueError('No tuning candidates')
    seedset = None
    summaries = []
    for candidate in candidates:
        trials = candidate['trials']
        seeds = {trial['seed'] for trial in trials}
        if len(trials) != 5 or len(seeds) != 5 or (seedset is not None and seeds != seedset):
            raise ValueError('Every configuration requires the same five distinct trials')
        seedset = seeds
        import numpy as np
        values = np.array([[t['final_accuracy'],t['rop']] for t in trials])
        if not np.isfinite(values).all():
            raise ValueError('Non-finite tuning metrics')
        summaries.append({'parameters': candidate['parameters'], 'mean_final_accuracy': float(values[:,0].mean()), 'mean_rop': float(values[:,1].mean())})
    key = lambda row: json.dumps(row['parameters'], sort_keys=True)
    ranked = sorted(summaries, key=lambda row:(-row['mean_final_accuracy'],key(row)))
    count = max(1, math.ceil(.025*len(ranked)))
    selected = sorted(ranked[:count], key=lambda row:(-row['mean_rop'],-row['mean_final_accuracy'],key(row)))[0]
    return {'selected': selected, 'top_count': count, 'rounding_policy': 'ceil; ties by parameter JSON', 'configurations': summaries}


def _load_seed_bundle(args, seed):
    """Load and split one seed bundle once so tuning configs reuse identical splits.

    Parameters: base CLI args mapping and integer seed.
    Returns: validated DatasetBundle for that seed.
    """
    if args['dataset'] == 'synthetic':
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=args.get('samples', 300), n_features=20, n_informative=8, random_state=seed
        )
        return split_data(X, 2 * y - 1, seed=seed, metadata={'dataset': 'synthetic', 'development_subset': True})
    return load_dataset(
        args['dataset'], seed=seed, source_file=args.get('sources'), max_samples=args.get('max_samples')
    )


def _run_seed_chunk(task):
    """Evaluate a slice of parameter settings on one pre-loaded seed bundle.

    Parameters: tuple (args, seed, indexed_settings, out_dir).
    Returns: list of (config_index, trial_metrics_dict).
    """
    args, seed, indexed_settings, out_dir = task
    bundle = _load_seed_bundle(args, seed)
    results = []
    for i, parameters in indexed_settings:
        config = dict(args, **parameters)
        config.update(seed=seed, experiment='base-tuning')
        run_dir = Path(out_dir) / f'config-{i:03d}' / f'seed-{seed}'
        run_experiment(config, run_dir, bundle=bundle)
        metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))['baseline']
        results.append((i, dict(seed=seed, final_accuracy=metrics['final_accuracy'], rop=metrics['rop'])))
    return results


def main():
    """Evaluate the full configured learner grid on five trials and save selection."""
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    p = parser()
    p.add_argument('--grid', default=str(ROOT/'configs/full_scope.json'))
    p.add_argument('--jobs', type=int, default=1)
    args = vars(p.parse_args())
    jobs = args.pop('jobs')
    if jobs < 1:
        p.error('--jobs must be at least 1')
    if args['include_wrs'] or args['noise_rate']:
        p.error('Tune clean base learners only')
    out = args.pop('output')
    grid = json.loads(Path(args.pop('grid')).read_text(encoding='utf-8'))
    settings = ([{'C_err': c} for c in grid['tuning']['pac_C_err']] if args['learner']=='pac' else
                [{'eta':eta,'lambda_':lam} for eta,lam in product(grid['tuning']['fsol_eta'],grid['tuning']['fsol_lambda'])])
    # Refuse overwrite before running expensive jobs.
    out.mkdir(parents=True, exist_ok=False)
    for i in range(len(settings)):
        (out / f'config-{i:03d}').mkdir()
    indexed = list(enumerate(settings))
    seeds = grid['seeds']
    n_chunks = 1 if jobs == 1 else min(len(indexed), max(1, math.ceil(jobs / len(seeds))))
    chunk_size = math.ceil(len(indexed) / n_chunks)
    tasks = [
        (args, seed, indexed[start:start + chunk_size], str(out))
        for seed in seeds
        for start in range(0, len(indexed), chunk_size)
    ]
    trials_by_cfg = {i: [] for i in range(len(settings))}
    if jobs == 1:
        batches = [_run_seed_chunk(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as pool:
            batches = list(pool.map(_run_seed_chunk, tasks))
    for batch in batches:
        for i, trial in batch:
            trials_by_cfg[i].append(trial)
    candidates = [{'parameters': parameters, 'trials': trials_by_cfg[i]} for i, parameters in indexed]
    write_json(out/'selection.json', select_parameters(candidates))


if __name__ == '__main__':
    main()

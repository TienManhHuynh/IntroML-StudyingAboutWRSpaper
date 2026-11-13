"""Appendix B.2 base-learner grid search, with an explicit top-2.5% rounding policy."""
from itertools import product
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_reproduction import parser, run_experiment, write_json


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


def main():
    """Evaluate the full configured learner grid on five trials and save selection."""
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env', override=False)
    p = parser()
    p.add_argument('--grid', default=str(ROOT/'configs/full_scope.json'))
    args = vars(p.parse_args())
    if args['include_wrs'] or args['noise_rate']:
        p.error('Tune clean base learners only')
    out = args.pop('output')
    grid = json.loads(Path(args.pop('grid')).read_text(encoding='utf-8'))
    settings = ([{'C_err': c} for c in grid['tuning']['pac_C_err']] if args['learner']=='pac' else
                [{'eta':eta,'lambda_':lam} for eta,lam in product(grid['tuning']['fsol_eta'],grid['tuning']['fsol_lambda'])])
    # Refuse overwrite before running expensive jobs.
    out.mkdir(parents=True, exist_ok=False)
    candidates = []
    for i, parameters in enumerate(settings):
        trials = []
        for seed in grid['seeds']:
            config = dict(args, **parameters)
            config.update(seed=seed, experiment='base-tuning')
            run_dir = out/f'config-{i:03d}'/f'seed-{seed}'
            run_experiment(config, run_dir)
            metrics = json.loads((run_dir/'metrics.json').read_text())['baseline']
            trials.append(dict(seed=seed, final_accuracy=metrics['final_accuracy'], rop=metrics['rop']))
        candidates.append({'parameters':parameters,'trials':trials})
    write_json(out/'selection.json', select_parameters(candidates))


if __name__ == '__main__':
    main()

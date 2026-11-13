"""Explicit W&B logging, with local evidence preserved if remote tracking fails."""
import json
import os
from pathlib import Path


def track_run(config, checkpoints, metrics, output, mode='disabled', project=None, entity=None):
    """Log one paired experiment to W&B and return URL only for online runs.

    Parameters: serializable config, checkpoint rows, metrics mapping, output directory,
    mode (disabled/offline/online), project and optional entity. No dataset is uploaded.
    Returns: online run URL, or None for disabled/offline. Errors propagate to the runner.
    """
    if mode not in ('disabled', 'offline', 'online'):
        raise ValueError('Invalid W&B mode')
    if mode == 'disabled':
        return None
    if not project:
        raise ValueError('A W&B project is required for tracking')
    root = Path(__file__).resolve().parents[2] / 'wandb'
    for variable, folder in (('WANDB_DIR','runs'),('WANDB_DATA_DIR','data'),('WANDB_CACHE_DIR','cache'),('WANDB_CONFIG_DIR','config')):
        path = root / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(path))
    import wandb
    run = wandb.init(project=project, entity=entity, mode=mode, config=config, job_type=config['experiment'], reinit='finish_previous')
    try:
        for step in sorted({row['step'] for row in checkpoints}):
            row_metrics = {'train_step': step}
            for row in (r for r in checkpoints if r['step'] == step):
                row_metrics[row['model'] + '/accuracy'] = row['accuracy']
                row_metrics[row['model'] + '/sparsity'] = row['sparsity']
            run.log(row_metrics)
        for model, summary in metrics.items():
            for key, value in summary.items():
                run.summary[model + '/' + key] = value
        artifact = wandb.Artifact('experiment-evidence-' + run.id, type='evaluation')
        for name in ('config.json', 'environment.json', 'checkpoints.csv', 'metrics.json', 'split-indices.npz'):
            artifact.add_file(str(Path(output) / name))
        run.log_artifact(artifact)
        url = run.url if mode == 'online' else None
        run.finish()
        return url
    except BaseException:
        run.finish(exit_code=1)
        raise

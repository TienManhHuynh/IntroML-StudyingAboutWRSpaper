"""TV3 data/metric and end-to-end evidence invariants, using small synthetic inputs."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.sparse import csr_matrix
import pytest
from src.data.loader import validate_features, split_data, training_label_noise
from src.evaluation.metrics import accuracy, sparsity, relative_oracle_performance, checkpoint_steps, deviation
from experiments.run_reproduction import run_experiment
from experiments.run_tuning import select_parameters
from experiments.summarize_results import summarize


def config(**kw):
    value=dict(dataset='synthetic',learner='pac',seed=42,C_err=1,eta=1,lambda_=.01,samples=80,
               checkpoints=10,include_wrs=False,noise_rate=0,noise_seed=100042,experiment='test',wandb_mode='disabled')
    value.update(kw); return value


def test_metric_oracle_and_negative_rop():
    assert accuracy([1,-1,1],[1,1,1])==pytest.approx(2/3)
    assert sparsity([0,1,0,-1])==.5
    assert relative_oracle_performance([.8,.5,.9],[.85,.85,.95])==pytest.approx(-.05)
    assert relative_oracle_performance([.8,.5,.9],[.8,.5,.9])==pytest.approx(.1)
    assert deviation(0,.1)['relative_percent'] is None
    with pytest.raises(ValueError): relative_oracle_performance([.5],[.5,.6])
    with pytest.raises(ValueError): accuracy([],[])


def test_checkpoint_boundaries():
    steps=checkpoint_steps(3,200)
    assert steps.tolist()==[0,1,2,3]
    assert len(checkpoint_steps(1000))==201


@pytest.mark.parametrize('as_sparse',[False,True])
def test_nonfinite_and_label_guard(as_sparse):
    X=np.array([[1.,0],[float('nan'),1]])
    with pytest.raises(ValueError): validate_features(csr_matrix(X) if as_sparse else X,[1,-1])
    with pytest.raises(ValueError): validate_features(np.ones((2,2)),[0,1])


def test_split_noise_determinism_no_mutation():
    X=csr_matrix(np.arange(120).reshape(40,3)); y=np.tile([-1,1],20)
    a=split_data(X,y,42); b=split_data(X,y,42)
    np.testing.assert_array_equal(a.train_indices,b.train_indices)
    assert not set(a.train_indices)&set(a.test_indices)
    old=a.y_test.copy(); source=a.y_train.copy()
    noisy,mask=training_label_noise(source,1,9)
    np.testing.assert_array_equal(noisy,-source); np.testing.assert_array_equal(a.y_test,old)
    np.testing.assert_array_equal(source,a.y_train)
    assert mask.all()


def test_runner_evidence_and_immutability(tmp_path):
    out=tmp_path/'run'; state=run_experiment(config(),out)
    assert state['status']=='completed_local' and not state['reportable']
    assert json.loads((out/'metrics.json').read_text())['baseline']['rop']>=0
    assert (out/'environment.json').exists() and (out/'split-indices.npz').exists()
    with pytest.raises(FileExistsError): run_experiment(config(),out)
    second=tmp_path/'run2'; run_experiment(config(),second)
    assert (out/'checkpoints.csv').read_bytes()==(second/'checkpoints.csv').read_bytes()
    with pytest.raises(ValueError): summarize([out,second],tmp_path/'summary')


def test_missing_wat_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='WAT implementation pending'):
        run_experiment(config(include_wrs=True),tmp_path/'run')
    assert not (tmp_path/'run').exists()


def test_tracking_failure_preserves_local_evidence(tmp_path):
    out=tmp_path/'run'
    with patch('experiments.run_reproduction.track_run',side_effect=RuntimeError('network')):
        with pytest.raises(RuntimeError): run_experiment(config(),out)
    assert json.loads((out/'run.json').read_text())['status']=='failed'
    assert (out/'metrics.json').exists()


def test_tuning_requires_five_matched_seeds():
    values=[{'parameters':{'C_err':i},'trials':[{'seed':s,'final_accuracy':.7+i*.001,'rop':i*.01} for s in range(5)]} for i in range(41)]
    for trial in values[39]['trials']:
        trial['rop']=10  # Lower accuracy within the top-two pool, but higher ROP.
    selected=select_parameters(values)
    assert selected['top_count']==2 and selected['selected']['parameters']=={'C_err':39}
    values[0]['trials'].pop()
    with pytest.raises(ValueError): select_parameters(values)


def test_injected_wat_factory_pairs_models(tmp_path, monkeypatch):
    import sys, types
    module=types.ModuleType('test_wat_contract')
    received={}
    # Contract double only: deliberately identity, not a WAT implementation.
    def factory(**kwargs):
        received.update(kwargs)
        return kwargs['base_learner']
    module.factory=factory
    monkeypatch.setitem(sys.modules,'test_wat_contract',module)
    out=tmp_path/'paired'
    run_experiment(config(include_wrs=True,wat_factory='test_wat_contract:factory',K=4,weighting='standard',averaging='simple',zeroing=False),out)
    metrics=json.loads((out/'metrics.json').read_text())
    assert metrics['baseline']['final_accuracy']==metrics['wrs']['final_accuracy']
    assert metrics['baseline']['rop']==metrics['wrs']['rop']
    assert received['K']==4 and received['seed']==200042


def test_hf_text_source_train_only_vocabulary(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.data.loader import load_dataset
    path=tmp_path/'text.parquet'
    pq.write_table(pa.table({'sentence':[f'common unique{i}' for i in range(30)],'label':[i%2 for i in range(30)]}),path)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', lambda **kwargs:str(path))
    spec=tmp_path/'sources.json'
    spec.write_text(json.dumps({'fixture':{'repo_id':'fixture/test','revision':'a'*40,'filename':'text.parquet',
                         'format':'text_parquet','text_column':'sentence','label_column':'label','label_mapping':{'0':-1,'1':1}}}))
    bundle=load_dataset('fixture',source_file=spec)
    assert bundle.X_train.shape[1]==len(bundle.y_train)+1
    assert (bundle.X_test.getnnz(axis=1)==1).all()  # unique test tokens never entered vocabulary
    assert bundle.metadata['revision']=='a'*40


def test_hf_sparse_npz_source(tmp_path, monkeypatch):
    from src.data.loader import load_dataset
    X = csr_matrix(np.eye(20, 5))
    y = np.tile([-1, 1], 10)
    path = tmp_path / 'sparse.npz'
    np.savez_compressed(path, data=X.data, indices=X.indices, indptr=X.indptr, shape=X.shape, y=y)
    monkeypatch.setattr('huggingface_hub.hf_hub_download', lambda **kwargs: str(path))
    spec = tmp_path / 'sources.json'
    spec.write_text(json.dumps({'fixture': {'repo_id': 'fixture/sparse', 'revision': 'b'*40, 'filename': 'sparse.npz',
                         'format': 'sparse_npz', 'label_mapping': {'-1': -1, '1': 1}}}))
    bundle = load_dataset('fixture', source_file=spec)
    assert bundle.X_train.shape == (14, 5) and bundle.X_test.shape == (6, 5)
    assert bundle.metadata['revision'] == 'b' * 40


def test_tuning_parallel_matches_sequential(tmp_path, monkeypatch):
    import sys
    from experiments import run_tuning
    grid = tmp_path / 'grid.json'
    grid.write_text(json.dumps({
        'seeds': [42, 43, 44, 45, 46],
        'tuning': {'pac_C_err': [0.1, 1.0], 'fsol_eta': [1.0], 'fsol_lambda': [0.01]},
    }))
    out_seq = tmp_path / 'seq'
    out_par = tmp_path / 'par'
    for out, jobs in ((out_seq, '1'), (out_par, '2')):
        monkeypatch.setattr(sys, 'argv', [
            'run_tuning.py', '--dataset', 'synthetic', '--samples', '60', '--checkpoints', '10',
            '--learner', 'pac', '--grid', str(grid), '--jobs', jobs, '--output', str(out),
        ])
        run_tuning.main()
    assert json.loads((out_seq / 'selection.json').read_text()) == json.loads((out_par / 'selection.json').read_text())


def test_ablation_wrappers_and_runner(tmp_path):
    from experiments.run_ablations import (
        ExponentialAverageEnsemble,
        MovingAverageEnsemble,
        TopKEnsemble,
        run_ablation,
    )
    from src.models.pac import PAC

    topk = TopKEnsemble(PAC(2, C_err=0.5), K=2, averaging='weighted')
    topk.update([1., 0.], 1)
    topk.update([2., 0.], 1)
    topk.update([2., 0.], 1)
    topk.update([-2., 0.], -1)
    assert len(topk._heap) <= 2 and topk.get_weights().shape == (2,)

    ma = MovingAverageEnsemble(PAC(2, C_err=0.5), K=2)
    ma.update([1., 0.], 1)
    w1 = ma.base_learner.get_weights()
    ma.update([0., 1.], 1)
    w2 = ma.base_learner.get_weights()
    np.testing.assert_allclose(ma.get_weights(), 0.5 * (w1 + w2))

    ema = ExponentialAverageEnsemble(PAC(2, C_err=0.5), gamma=0.9)
    ema.update([1., 0.], 1)
    np.testing.assert_allclose(ema.get_weights(), 0.9 * ema.base_learner.get_weights())

    for method in ('topk', 'moving_average', 'exponential_average'):
        out = tmp_path / method
        state = run_ablation(dict(
            dataset='synthetic', learner='pac', method=method, seed=42, C_err=1.0,
            eta=1.0, lambda_=0.01, K=4, averaging='simple', gamma=0.9, samples=60,
            checkpoints=10, experiment='test-ablation', wandb_mode='disabled',
        ), out)
        assert state['status'] == 'completed_local'
        metrics = json.loads((out / 'metrics.json').read_text())
        assert 'baseline' in metrics and method in metrics

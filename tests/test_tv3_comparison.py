import csv
import pytest
from experiments.compare_reference import compare
from src.evaluation.statistics import paired_dataset_test


def test_paired_keys_and_all_zero():
    result=paired_dataset_test({'a':.1,'b':.2},{'a':.1,'b':.2})
    assert result['pvalue']==1 and not result['full_16']
    with pytest.raises(ValueError): paired_dataset_test({'a':1},{'b':1})


def test_comparison_zero_denominator_and_unmatched(tmp_path):
    fields=['dataset','learner','model','seed','config_id','source','final_accuracy','rop','final_sparsity']
    row=dict(zip(fields,['d','pac','baseline','42','verified-config','source',.8,0,.5]))
    for name in ('ref','team'):
        with (tmp_path/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerow(row)
    assert compare(tmp_path/'ref.csv',tmp_path/'team.csv',tmp_path/'out.csv')['matched_trials']==1
    rows=list(csv.DictReader((tmp_path/'out.csv').open()))
    assert next(r for r in rows if r['metric']=='rop')['relative_percent']==''
    with pytest.raises(FileExistsError): compare(tmp_path/'ref.csv',tmp_path/'team.csv',tmp_path/'out.csv')

"""Paired dataset-level statistics; run-level replicates must be averaged first."""
import numpy as np
from scipy.stats import wilcoxon


def paired_dataset_test(baseline, candidate, alternative='two-sided'):
    """Return Wilcoxon statistics on matching dataset->trial-mean values, with win count.

    Lower is better (ROP). Same dataset keys required. Tests with fewer than 16 datasets
    are explicitly partial; all-zero differences return p=1. Zero policy is wilcox.
    """
    if set(baseline)!=set(candidate) or not baseline:
        raise ValueError('Nonempty paired dataset keys required')
    names=sorted(baseline)
    left=np.array([baseline[n] for n in names],dtype=float)
    right=np.array([candidate[n] for n in names],dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Finite per-dataset mean values required')
    if alternative not in ('two-sided','less','greater'):
        raise ValueError('Invalid Wilcoxon alternative')
    differences=left-right
    if np.all(differences==0):
        statistic,pvalue=0.,1.
    else:
        result=wilcoxon(left,right,alternative=alternative,zero_method='wilcox',method='auto')
        statistic,pvalue=float(result.statistic),float(result.pvalue)
    return {'datasets':names,'n_datasets':len(names),'full_16':len(names)==16,
            'candidate_wins_lower_is_better':int((right<left).sum()),'statistic':statistic,'pvalue':pvalue,
            'alternative':alternative,'zero_method':'wilcox','difference':'baseline - candidate'}

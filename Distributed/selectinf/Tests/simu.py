import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import namedtuple
import os

from ..distributed_lasso import multisplit_lasso as L

from .instance import gaussian_instance

Metrics = namedtuple('Metrics', ['precision', 'recall', 'f1'])

def get_metrics(beta, signs, accept):
    reject = ~accept
    true_positive = np.sum(reject * (beta[signs] != 0))
    true_negative = np.sum(accept * (beta[signs] == 0))
    all_pred_positive = np.sum(reject)  # number of predicted positives
    if all_pred_positive == 0:
        precision = np.nan
    else:
        precision = true_positive / all_pred_positive  # ppv
    all_positive = np.sum(beta[signs] != 0)  # number of positives that are selected
    if all_positive == 0:
        recall = np.nan
    else:
        recall = true_positive / all_positive
    
    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2 / (1 / precision + 1 / recall)
    return Metrics(precision, recall, f1_score)


def test(seedn,
         n=1000,
         p=500,
         signal_fac=1.2,
         s=5,
         sigma=3,
         rho=0.9,
         targets='selected'):
    np.random.seed(seedn)
    inst = gaussian_instance
    coverages = {}
    lengths = {}
    metrics = {}

    while True:  # run until we get some selection

        signal = np.sqrt(signal_fac * 2 * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        n, p = X.shape

        sigma_ = np.std(Y)

        if n > (2 * p):
            full_dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            full_dispersion = sigma_ ** 2

        nK = 3

        feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ for i in range(nK - 1)}
        # proportion = [1 / 2, 1 / 3]
        proportion = .3

        selector = L.gaussian(X,
                              Y,
                              feature_weights,
                              proportion,
                              estimate_dispersion=True,
                              sample_with_replacement=True)

        signs = selector.fit()
        nonzero = signs != 0
        # print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            selector.setup_inference(dispersion=None)

            target_spec = selector.selected_targets()

            result = selector.inference(target_spec,
                                        level=0.90)

            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])

            coverage = (intervals[:, 0] < beta_target) * (intervals[:, 1] > beta_target)

            length = intervals[:, 1] - intervals[:, 0]

            # print("check coverage + lengths ", np.mean(coverage), np.mean(length))
            coverages['dist_carving'] = coverage
            lengths['dist_carving'] = length

            accept = (intervals[:, 0] < 0) * (intervals[:, 1] > 0)
            metrics['dist_carving'] = get_metrics(beta, signs, accept)

            # splitting
            holdout_idx = np.sum(selector._selection_idx, 1) == 0
            X_holdout = X[holdout_idx][:, signs]
            Y_holdout = Y[holdout_idx]
            X_cov = np.linalg.inv(X_holdout.T @ X_holdout)
            splitting_estimator = X_cov @ X_holdout.T @ Y_holdout
            sds = np.sqrt(np.diag(X_cov)) * sigma
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))  # ground truth
            coverages['splitting'] = abs(splitting_estimator - beta_target) < sds * norm.ppf(0.95)
            lengths['splitting'] = sds * 2 * norm.ppf(0.95)

            accept = abs(splitting_estimator) < sds * norm.ppf(0.95)
            metrics['splitting'] = get_metrics(beta, signs, accept)

            # naive intervals
            X_full_cov = np.linalg.inv(X[:, signs].T @ X[:, signs])
            full_sds = np.sqrt(np.diag(X_full_cov)) * sigma
            beta_ols = X_full_cov @ X[:, signs].T @ Y
            coverages['naive'] = abs(beta_ols - beta_target) < full_sds * norm.ppf(0.95)
            lengths['naive'] = full_sds * 2 * norm.ppf(0.95)

            accept = abs(beta_ols) < full_sds * norm.ppf(0.95)
            metrics['naive'] = get_metrics(beta, signs, accept)

            return coverages, lengths, metrics


def main(nsim=500):
    print_every = 20
    cover_ = {t: [] for t in ['dist_carving', 'splitting', 'naive']}
    len_ = {t: [] for t in ['dist_carving', 'splitting', 'naive']}
    f1_ = {t: [] for t in ['dist_carving', 'splitting', 'naive']}
    precision_ = {t: [] for t in ['dist_carving', 'splitting', 'naive']}
    recall_ = {t: [] for t in ['dist_carving', 'splitting', 'naive']}

    for i in range(nsim):
        coverages, lengths, metrics = test(seedn=i, n=3000, p=500, sigma=1.)
        [cover_[key].extend(coverages[key]) for key in coverages.keys()]
        [len_[key].extend(lengths[key]) for key in lengths.keys()]
        [f1_[key].extend([metrics[key].f1]) for key in metrics.keys()]
        [precision_[key].extend([metrics[key].precision]) for key in metrics.keys()]
        [recall_[key].extend([metrics[key].recall]) for key in metrics.keys()]

        if (i + 1) % print_every == 0 or i == nsim - 1:
            df = pd.DataFrame(columns=cover_.keys(), index=['coverage', 'length', 'f1', 'precision', 'recall'])
            for key in cover_.keys():
                df[key] = [np.nanmean(cover_[key]), np.nanmean(len_[key]), np.nanmean(f1_[key]), np.nanmean(precision_[key]), np.nanmean(recall_[key])]
            print("========= Progress {:.1f}% =========".format(100 * (i + 1) / nsim))
            print(df)
        
    root_dir = 'Distributed/selectinf/Tests/results'
    os.makedirs(root_dir, exist_ok=True)
    df.to_csv(os.path.join(root_dir, 'results_with_replace.csv'))

if __name__ == "__main__":
    main(nsim=500)

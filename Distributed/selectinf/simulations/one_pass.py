import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import namedtuple
import os
from ..distributed_lasso import multisplit_lasso as L
from ..Tests.instance import gaussian_instance
from .metrics import get_metrics
import argparse
import pickle
import logging
import itertools


def run(seedn,
        n=1000,
        p=500,
        signal_fac=1.2,
        s=5,
        sigma=3,
        rho=0.9,
        nK=3,  # total number of machines, including holdout set
        proportion=None,
        level=0.9,
        weight_fac=1.,
        sample_with_replacement=False,
        targets='selected'):
    np.random.seed(seedn)
    inst = gaussian_instance
    coverages = {}  # cover the target beta in selected/saturated view
    lengths = {}
    metrics = {}

    signal = np.sqrt(signal_fac * 2 * np.log(p))
    X, Y, beta = inst(n=n,
                        p=p,
                        signal=signal,
                        s=s,
                        equicorrelated=False,
                        rho=rho,
                        sigma=sigma,
                        random_signs=True)[:3]
    true_signal = beta != 0
    idx = np.arange(p)
    n, p = X.shape

    sigma_ = np.std(Y)

    if n > (2 * p):
        full_dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        full_dispersion = sigma_ ** 2

    feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ * weight_fac for i in range(nK - 1)}
    if proportion is None:
        proportion = np.ones(nK - 1) / (nK)

    selector = L.gaussian(X,
                            Y,
                            feature_weights,
                            proportion,
                            estimate_dispersion=True,
                            sample_with_replacement=sample_with_replacement)

    signs = selector.fit()
    nonzero = signs != 0
    screening = sum(true_signal * nonzero) == sum(true_signal)
    # print("dimensions", n, p, nonzero.sum())
    if nonzero.sum() == 0:
        return None, None, None, None

    if nonzero.sum() > 0:
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))  

        selector.setup_inference(dispersion=sigma) # dispersion=sigma

        target_spec = selector.selected_targets()

        result = selector.inference(target_spec,
                                    level=level)

        intervals = np.asarray(result[['lower_confidence',
                                        'upper_confidence']])

        coverage = (intervals[:, 0] < beta_target) * (intervals[:, 1] > beta_target)

        length = intervals[:, 1] - intervals[:, 0]

        # print("check coverage + lengths ", np.mean(coverage), np.mean(length))
        coverages['dist_carving'] = coverage
        lengths['dist_carving'] = length

        accept = (intervals[:, 0] < 0) * (intervals[:, 1] > 0)
        metrics['dist_carving'] = get_metrics(beta[nonzero], accept)

        # splitting
        holdout_idx = np.sum(selector._selection_idx, 1) == 0
        X_holdout = X[holdout_idx][:, signs]
        Y_holdout = Y[holdout_idx]
        X_cov = np.linalg.inv(X_holdout.T @ X_holdout)
        splitting_estimator = X_cov @ X_holdout.T @ Y_holdout
        sds = np.sqrt(np.diag(X_cov)) * sigma
    
        coverages['splitting'] = abs(splitting_estimator - beta_target) < sds * norm.ppf(.5 + level / 2)
        lengths['splitting'] = sds * 2 * norm.ppf(.5 + level / 2)

        accept = abs(splitting_estimator) < sds * norm.ppf(.5 + level / 2)
        metrics['splitting'] = get_metrics(beta[nonzero], accept)

        # naive intervals
        X_full_cov = np.linalg.inv(X[:, signs].T @ X[:, signs])
        full_sds = np.sqrt(np.diag(X_full_cov)) * sigma
        beta_ols = X_full_cov @ X[:, signs].T @ Y
        coverages['naive'] = abs(beta_ols - beta_target) < full_sds * norm.ppf(.5 + level / 2)
        lengths['naive'] = full_sds * 2 * norm.ppf(.5 + level / 2)

        accept = abs(beta_ols) < full_sds * norm.ppf(.5 + level / 2)
        metrics['naive'] = get_metrics(beta[nonzero], accept)

        return coverages, lengths, metrics, screening


def main(sample_with_replacement, nK, n0, n1, p, s, signal_fac, weight_fac):
    print(f"Starting simulation with K={nK}, n0={n0}, n1={n1}, p={p}, s={s}, signal_fac={signal_fac}, weight_fac={weight_fac}")
    nsim = 500
    print_every = 50
    methods = ['dist_carving', 'splitting', 'naive']
    coverages_ = {t: [] for t in methods}
    lengths_ = {t: [] for t in methods}
    metrics_ = {t: [] for t in methods}
    screening_ = []

    # fcr_ = {t: [] for t in methods}
    
    if sample_with_replacement:
        proportion = .5
        n = 2000
    else:
        n = (nK - 1) * n1 + n0
        proportion = np.ones(nK - 1) * (n1 / n)

    for i in range(nsim):
        coverages, lengths, metrics, screening = run(seedn=i, n=n, p=p, nK=nK, sigma=1., signal_fac=signal_fac, rho=0.5, s=s, proportion=proportion, sample_with_replacement=sample_with_replacement, weight_fac=weight_fac)
        if coverages is not None:
            methods = coverages.keys()
            [coverages_[key].append(coverages[key]) for key in methods]
            [lengths_[key].append(lengths[key]) for key in methods]
            [metrics_[key].append([metrics[key]]) for key in methods]
            screening_.append(screening)

        if (i + 1) % print_every == 0 or i == nsim - 1:
            mean_cover = {}
            for key in methods:
                mean_cover[key] = [np.mean([i for j in coverages_[key] for i in j]), np.mean([i for j in lengths_[key] for i in j])]
            print("========= Progress {:.1f}% =========".format(100 * (i + 1) / nsim))
            print(pd.DataFrame(mean_cover, index=['coverage', 'length']))
        
    root_dir = 'Distributed/selectinf/simulations/results'
    os.makedirs(root_dir, exist_ok=True)
    filename = f"K_{nK}_n0_{n0}_n1_{n1}_p_{p}_s_{s}_signal_{signal_fac}_weight_{weight_fac}"
    if sample_with_replacement:
        filename = os.path.join(root_dir, f'one_pass_with_replace_{filename}.pkl')
    else:
        filename = os.path.join(root_dir, f'one_pass_without_replace_{filename}.pkl')
    all_results = {'coverages': coverages_, 'lengths': lengths_, 'metrics': metrics_, 'screening': screening_}
    with open(filename, 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_replace', '-wr', default=False, action='store_true')
    parser.add_argument('--p', default=100, type=int)
    parser.add_argument('--jobid', default=0, type=int)
    args = parser.parse_args()

    K_list = [3, 5, 7]
    signals = np.linspace(0.5, 2, num=6)
    n0_list = [2000, 4000, 6000]
    s_list = [5, 10, 15]
    weight_facs = [1.5, 2.]
    # len(K_list) * len(signals) * len(n0_list) * len(s_list) * len(weight_facs)

    i = 0
    for nK, signal_fac, n0, s, weight_fac in itertools.product(K_list, signals, n0_list, s_list, weight_facs):
        if i == args.jobid:
            n1 = (10000 - n0) // (nK - 1)  # fix total n to be 10000
            main(args.w_replace, nK, n0, n1, args.p, s, signal_fac, weight_fac)
        i += 1




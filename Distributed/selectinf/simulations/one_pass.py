import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import namedtuple
import os
from ..distributed_lasso import multisplit_lasso as L
from ..Tests.instance import gaussian_instance, logistic_instance
from .metrics import get_metrics
import argparse
import pickle
import logging
import itertools
import regreg.api as rr
from ..Utils.base import restricted_estimator
from sklearn.linear_model import LogisticRegression

def solve_target_restricted(linpred, X, active):

    def pi(x):
        return 1 / (1 + np.exp(-x))

    Y_mean = pi(linpred)
    n = X.shape[0]

    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))

    _beta_unpenalized = restricted_estimator(loglike,
                                             active)
    return _beta_unpenalized

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
        logistic=False,
        sample_with_replacement=False,
        targets='selected',
        n_tuning=0,
        thre_agg=0):
    np.random.seed(seedn)
    
    coverages = {}  # cover the target beta in selected/saturated view
    lengths = {}
    metrics = {}

    signal = np.sqrt(signal_fac * 2 * np.log(p))
    if not logistic:
        inst = gaussian_instance
        X, Y, beta = inst(n=n+n_tuning,
                            p=p,
                            signal=signal,
                            s=s,
                            equicorrelated=False,
                            rho=rho,
                            sigma=sigma,
                            random_signs=True)[:3]
        if n_tuning > 0:
            X_tune = X[n:]  # used for tuning lambda
            Y_tune = Y[n:]
            X = X[:n]
            Y = Y[:n]
        sigma_ = np.std(Y)
        if n_tuning > 0:
            weight_facs = np.linspace(.5, 5, 10) 
            min_mse = np.Inf
            best_weight_fac = .5
            for weight_fac in weight_facs:
                feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ * weight_fac for i in range(nK - 1)}
                selector_ = L.gaussian(X,
                                        Y,
                                        feature_weights,
                                        proportion,
                                        estimate_dispersion=True,
                                        sample_with_replacement=sample_with_replacement)
                
                signs_ = selector_.fit(thre_agg=thre_agg)
                mse = np.linalg.norm(Y_tune - X_tune @ selector_._beta_full)**2
                if mse < min_mse:
                    min_mse = mse
                    best_weight_fac = weight_fac
                    selector = selector_
                    signs = signs_
            print(best_weight_fac)
    else:
        inst = logistic_instance
        X, Y, beta = inst(n=n+n_tuning,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=False,
                              rho=rho,
                              random_signs=True)[:3]
        if n_tuning > 0:
            X_tune = X[n:]  # used for tuning lambda
            Y_tune = Y[n:]
            X = X[:n]
            Y = Y[:n]
        linpred = X.dot(beta)
        sigma_ = 1
        feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) for i in range(nK - 1)}
        selector = L.logistic(X, Y, feature_weights, proportion)
        if n_tuning > 0:
            weight_facs = np.linspace(0.1, 1, num=10)
            max_loglik = -np.Inf
            best_weight_fac = weight_facs[0]
            for weight_fac in weight_facs:
                feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * weight_fac for i in range(nK - 1)}
                selector_ = L.logistic(X, Y, feature_weights, proportion)
                signs_ = selector_.fit()
                pi = X_tune @ selector_._beta_full
                pi = 1 / (1 + np.exp(-pi))
                loglik = np.sum(Y_tune * np.log(pi) + (1 - Y_tune) * np.log(1 - pi))
                if loglik > max_loglik:
                    max_loglik = loglik
                    best_weight_fac = weight_fac
                    selector = selector_
                    signs = signs_
            print(best_weight_fac)

    true_signal = beta != 0
    nonzero = signs != 0
    print("threshold:", thre_agg)
    print("dimensions", n, p, nonzero.sum())

    screening = sum(true_signal * nonzero) == sum(true_signal)
    # print("dimensions", n, p, nonzero.sum())
    if nonzero.sum() == 0:
        return None, None, None, None
    
    if nonzero.sum() > 0:
        if not logistic:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))  
            selector.setup_inference(dispersion=None) # dispersion=sigma
        else:
            beta_target = solve_target_restricted(linpred, X, selector.overall)
            selector.setup_inference(dispersion=1.)

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
        if not logistic:
            X_cov = np.linalg.inv(X_holdout.T @ X_holdout)
            splitting_estimator = X_cov @ X_holdout.T @ Y_holdout
            sds = np.sqrt(np.diag(X_cov)) * sigma

        else:
            ll = LogisticRegression(fit_intercept=False, penalty='none')
            ll.fit(X_holdout, Y_holdout)
            splitting_estimator = ll.coef_.reshape(-1)
            pi = 1 / (1 + np.exp(-X_holdout @ splitting_estimator))
            W = np.diag(pi * (1 - pi))
            hess = X_holdout.T @ W @ X_holdout
            cov = np.linalg.inv(hess)
            sds = np.sqrt(np.diag(cov))
    
        coverages['splitting'] = abs(splitting_estimator - beta_target) < sds * norm.ppf(.5 + level / 2)
        lengths['splitting'] = sds * 2 * norm.ppf(.5 + level / 2)

        accept = abs(splitting_estimator) < sds * norm.ppf(.5 + level / 2)
        metrics['splitting'] = get_metrics(beta[nonzero], accept)

        # naive intervals
        if not logistic:
            X_full_cov = np.linalg.inv(X[:, signs].T @ X[:, signs])
            full_sds = np.sqrt(np.diag(X_full_cov)) * sigma
            beta_ols = X_full_cov @ X[:, signs].T @ Y
        else:
            ll = LogisticRegression(fit_intercept=False, penalty='none')
            ll.fit(X[:, signs], Y)
            beta_ols = ll.coef_.reshape(-1)
            pi = 1 / (1 + np.exp(-X[:, signs] @ beta_ols))
            W = np.diag(pi * (1 - pi))
            hess = X[:, signs].T @ W @ X[:, signs]
            cov = np.linalg.inv(hess)
            full_sds = np.sqrt(np.diag(cov))

        coverages['naive'] = abs(beta_ols - beta_target) < full_sds * norm.ppf(.5 + level / 2)
        lengths['naive'] = full_sds * 2 * norm.ppf(.5 + level / 2)

        accept = abs(beta_ols) < full_sds * norm.ppf(.5 + level / 2)
        metrics['naive'] = get_metrics(beta[nonzero], accept)

        return coverages, lengths, metrics, screening


def main(seed, sample_with_replacement, nK, n0, n1, p, s, signal_fac, weight_fac, logistic, n_tuning=0, root_dir='', thre_agg=0):
    print(f"Starting simulation with K={nK}, n0={n0}, n1={n1}, p={p}, s={s}, signal_fac={signal_fac}, weight_fac={weight_fac}")
    nsim = 1
    print_every = 1
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
        n = int((nK - 1) * n1 + n0)
        proportion = np.ones(nK - 1) * (n1 / n)
        print(n, p, proportion)

    for i in range(1):
        coverages, lengths, metrics, screening = run(seedn=seed, n=n, p=p, nK=nK, sigma=1., signal_fac=signal_fac, rho=0.9, s=s, proportion=proportion, sample_with_replacement=sample_with_replacement, weight_fac=weight_fac, logistic=logistic, n_tuning=n_tuning, thre_agg=thre_agg)
        if coverages is not None:
            methods = coverages.keys()
            [coverages_[key].append(coverages[key]) for key in methods]
            [lengths_[key].append(lengths[key]) for key in methods]
            [metrics_[key].append([metrics[key]]) for key in methods]
            screening_.append(screening)

        # if (i + 1) % print_every == 0 or i == nsim - 1:
        # mean_cover = {}
        # for key in methods:
        #     mean_cover[key] = [np.mean([i for j in coverages_[key] for i in j]), np.mean([i for j in lengths_[key] for i in j])]
        # print("========= Progress {:.1f}% =========".format(100 * (i + 1) / nsim))
        # print(pd.DataFrame(mean_cover, index=['coverage', 'length']))
        df_cover = pd.DataFrame(coverages).mean()
        df_len = pd.DataFrame(lengths).mean()
        print(df_cover)
        df_metrics = pd.DataFrame(metrics, index=['precision', 'recall', 'f1'])
        df = pd.concat([df_cover, df_len, df_metrics.T], 1)
        
        if root_dir == '':
            root_dir = 'Distributed/selectinf/simulations/results'
        os.makedirs(root_dir, exist_ok=True)
        if n_tuning > 0:
            filename = f"K_{nK}_n0_{n0}_n1_{n1}_p_{p}_s_{s}_signal_{signal_fac}_ntune_{n_tuning}_thre_{thre_agg}_seed_{seed}"
        else:
            filename = f"K_{nK}_n0_{n0}_n1_{n1}_p_{p}_s_{s}_signal_{signal_fac}_weight_{weight_fac}_thre_{thre_agg}_seed_{seed}"
        if logistic:
            filename += '_logistic'
        if sample_with_replacement:
            filename = os.path.join(root_dir, f'one_pass_with_replace_{filename}.csv')
        else:
            filename = os.path.join(root_dir, f'one_pass_without_replace_{filename}.csv')
        df.to_csv(filename)
        # all_results = {'coverages': coverages_, 'lengths': lengths_, 'metrics': metrics_, 'screening': screening_}
        # with open(filename, 'wb') as f:
            # pickle.dump(all_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_replace', '-wr', default=False, action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--root_dir', default='', type=str)
    parser.add_argument('--logistic', default=False, action='store_true')
    args = parser.parse_args()

    p = 100
    s = 5
    n0 = 1000
    # n0_list = [2000, 1000, 500, 250]
    # n1 = 2000
    weight_fac = 1
    # main(args.w_replace, K, n0, n1, p, s=5, signal_fac=.5, weight_fac=2.)

    K_list = [3, 5, 7, 9]
    nK = args.K
    # signals = np.linspace(0.5, 2, num=6)
    # signals = np.linspace(0.1, 0.9, 5)
    signal_fac = .1

    n1 = 8000 // (nK - 1)
    for thre_agg in range(nK):
        main(args.seed, args.w_replace, nK, n0, n1, p, s, signal_fac, weight_fac, args.logistic, 1000, root_dir=args.root_dir, thre_agg=thre_agg)
    
    # len(K_list) * len(signals) * len(n0_list) * len(s_list) * len(weight_facs)

    # i = 0
    # for nK, seed in itertools.product(K_list, np.arange(100)):
    #     if i == args.jobid:
    #         n1 = 8000 // (nK - 1)
    #         main(seed, args.w_replace, nK, n0, n1, args.p, s, signal_fac, weight_fac, args.logistic, 1000, root_dir=args.root_dir, aggregation_rule='majority')
    #     i += 1


import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from collections import namedtuple
import os
import argparse
import regreg.api as rr

from Distributed.selectinf.Utils.base import logistic_target_restricted, aggregate_shavegroups, get_metrics
from Distributed.selectinf.distributed_lasso import multisplit_lasso as L
from Distributed.selectinf.Tests.instance import gaussian_instance, logistic_instance, gaussian_grouped_instance


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
        weight_fac=.1,
        logistic=False,
        sample_with_replacement=False,
        n_tuning=0,
        one_per_group=False,
        rootdir=''):
    
    np.random.seed(seedn)
    
    coverages = {}  # cover the target beta in selected/saturated view
    lengths = {}
    metrics = {}

    signal = np.sqrt(signal_fac * 2 * np.log(p))
    if not logistic:
        inst = gaussian_instance
        if not one_per_group:
            X, Y, beta = inst(n=n+n_tuning, p=p, signal=signal, s=s, equicorrelated=False, rho=rho, sigma=sigma, random_signs=True)[:3]
        else:
            X, Y, beta= gaussian_grouped_instance(n=n+n_tuning, p=p, s=s, sigma=sigma, rho=rho, signal=signal, random_signs=True, equicorrelated=True, n_group=20, one_per_group=one_per_group)[:3]
        if n_tuning > 0:
            X_tune = X[n:]  # used for tuning lambda
            Y_tune = Y[n:]
            X = X[:n]
            Y = Y[:n]
        sigma_ = np.std(Y)
        feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ * weight_fac for i in range(nK - 1)}
        selector = L.gaussian(X, Y, feature_weights, proportion, estimate_dispersion=True, sample_with_replacement=sample_with_replacement)
        if one_per_group:
            aggregate=None
        else:
            aggregate = aggregate_shavegroups
        signs = selector.fit(aggregate=aggregate)
        # if n_tuning > 0:
        #     weight_facs = np.linspace(.5, 2, 50) 
        #     min_mse = np.Inf
        #     best_weight_fac = .5
        #     for weight_fac in weight_facs:
        #         feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ * weight_fac for i in range(nK - 1)}
        #         selector_ = L.gaussian(X, Y, feature_weights, proportion, estimate_dispersion=True, sample_with_replacement=sample_with_replacement)
                
        #         if one_per_group:
        #             aggregate=None
        #         else:
        #             aggregate = aggregate_shavegroups
        #         signs_ = selector_.fit(aggregate=aggregate)
        #         mse = np.linalg.norm(Y_tune - X_tune @ selector_._beta_full)**2
        #         if mse < min_mse:
        #             min_mse = mse
        #             best_weight_fac = weight_fac
        #             selector = selector_
        #             signs = signs_
        #     print("threhold =", thre_agg, 'selected', np.sum(selector.overall), 'variables')
        #     print('selected weight', best_weight_fac)
    else:
        inst = logistic_instance
        X, Y, beta = inst(n=n+n_tuning, p=p, signal=signal, s=s, equicorrelated=False, rho=rho, random_signs=True)[:3]
        if n_tuning > 0:
            X_tune = X[n:]  # used for tuning lambda
            Y_tune = Y[n:]
            X = X[:n]
            Y = Y[:n]
        linpred = X.dot(beta)
        sigma_ = 1
        feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * weight_fac for i in range(nK - 1)}
        selector = L.logistic(X, Y, feature_weights, proportion)
        signs = selector.fit()
        # if n_tuning > 0:
        #     weight_facs = np.linspace(0.1, 1, num=50)
        #     max_loglik = -np.Inf
        #     best_weight_fac = weight_facs[0]
        #     for weight_fac in weight_facs:
        #         feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * weight_fac for i in range(nK - 1)}
        #         selector_ = L.logistic(X, Y, feature_weights, proportion)
        #         signs_ = selector_.fit(thre_agg=thre_agg)
        #         pi = X_tune @ selector_._beta_full
        #         pi = 1 / (1 + np.exp(-pi))
        #         loglik = np.sum(Y_tune * np.log(pi) + (1 - Y_tune) * np.log(1 - pi))
        #         if loglik > max_loglik:
        #             max_loglik = loglik
        #             best_weight_fac = weight_fac
        #             selector = selector_
        #             signs = signs_
        #     print('selected weight', best_weight_fac)

    true_signal = beta != 0
    nonzero = signs != 0
    screening = sum(true_signal * nonzero) == sum(true_signal)
    if nonzero.sum() == 0:
        print("No variables selected")
        return
    
    if not logistic:
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))  
        selector.setup_inference(dispersion=None) # dispersion=sigma
    else:
        beta_target = logistic_target_restricted(linpred, X, selector.overall)
        selector.setup_inference(dispersion=1.)

    target_spec = selector.selected_targets()
    result = selector.inference(target_spec, level=level)
    intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

    coverages['dist_carving'] = (intervals[:, 0] < beta_target) * (intervals[:, 1] > beta_target)
    lengths['dist_carving'] = intervals[:, 1] - intervals[:, 0]

    rejects = np.zeros_like(beta, dtype=bool)
    rejects[nonzero] = (intervals[:, 1] < 0) + (intervals[:, 0] > 0)
    metrics['dist_carving'] = get_metrics(beta, rejects)

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
    rejects = np.zeros_like(beta, dtype=bool)
    rejects[nonzero] = abs(splitting_estimator) > sds * norm.ppf(.5 + level / 2)
    metrics['splitting'] = get_metrics(beta, rejects)

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
    rejects = np.zeros_like(beta, dtype=bool)
    rejects[nonzero] = abs(beta_ols) > full_sds * norm.ppf(.5 + level / 2)
    metrics['naive'] = get_metrics(beta, rejects)
    df_cover = pd.DataFrame(coverages).mean()
    df_len = pd.DataFrame(lengths).mean()
    # print(df_cover)
    df_metrics = pd.DataFrame(metrics, index=['TP', 'TN', 'FP', 'FN', 'F1', 'DOR'])
    df = pd.concat([df_cover, df_len], axis=1, keys=['coverage', 'length'])
    # df.columns = ['coverage', 'length']
    df = pd.concat([df, df_metrics.T], 1)
    df['num_selected'] = len(coverages['naive'])
    df['screening'] = screening
    print(df)
        
    if logistic:
        rootdir = os.path.join(rootdir, f"logistic_K_{nK}_n0_{n0}_n1_{n1}_p_{p}_s_{s}_signal_{signal_fac}_weightfac_{weight_fac}_ntune_{n_tuning}")
    else:
        rootdir = os.path.join(rootdir, f"gaussian_K_{nK}_n0_{n0}_n1_{n1}_p_{p}_s_{s}_signal_{signal_fac}_weightfac_{weight_fac}_ntune_{n_tuning}")
    os.makedirs(rootdir, exist_ok=True)

    filename = os.path.join(rootdir, f'seed_{seedn}.csv')
    df.to_csv(filename)
    print("Saved to", filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_replace', '-wr', default=False, action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--K', default=3, type=int)
    parser.add_argument('--rootdir', default='', type=str)
    parser.add_argument('--date', type=str, default='20240415')
    parser.add_argument('--logistic', default=False, action='store_true')
    parser.add_argument('--vary', default='K', type=str)
    args = parser.parse_args()

    p = 100
    s = 5
    n0 = 1000
    sample_with_replacement = args.w_replace
    if args.rootdir == '':
        rootdir = os.path.join('Distributed/selectinf/simulations/results/', args.date)
    else:
        rootdir = os.path.join(args.rootdir, args.date)
    
    weight_fac = 1
    K_list = [3, 5, 7, 9]
    nK = args.K
    if args.vary == 'K':
        signal_fac = 0.1
        for nK in K_list:
            n1 = 2000
            print('K =', nK)
            if sample_with_replacement:
                proportion = .5
                n = 2000
            else:
                n = 10000
            proportion = np.ones(nK - 1) * (n1 / n)
            run(seedn=args.seed, n=n, p=p, nK=nK, sigma=1., signal_fac=signal_fac, rho=0.9, s=s, proportion=proportion, sample_with_replacement=sample_with_replacement, weight_fac=weight_fac, logistic=args.logistic, n_tuning=0, one_per_group=False, rootdir=rootdir)
    
    elif args.vary == 'n0':
        n0_list = [2000, 1000, 500, 250]
        signal_fac = .5
        n1 = 2000
        nK = 4
        for n0 in n0_list:
            print('n0 =', n0)
            n = (nK - 1) * n1 + n0
            proportion = np.ones(nK - 1) * (n1 / n)
            run(seedn=args.seed, n=n, p=p, nK=nK, sigma=1., signal_fac=signal_fac, rho=0.9, s=s, proportion=proportion, sample_with_replacement=sample_with_replacement, weight_fac=weight_fac, logistic=args.logistic, n_tuning=0, one_per_group=False, rootdir=rootdir)
    
    elif args.vary == 'signal':
        signals = np.linspace(0.3, 0.9, num=4)
        if args.logistic:
            signals = [0.5, 1., 1.5, 2.]
            weight_fac = .5
        n1 = 4000
        n0 = 2000
        nK = 3
        n = (nK - 1) * n1 + n0
        proportion = np.ones(nK - 1) * (n1 / n)
        for signal_fac in signals:
            print('signal_fac =', signal_fac)
            run(seedn=args.seed, n=n, p=p, nK=nK, sigma=1., signal_fac=signal_fac, rho=0.9, s=s, proportion=proportion, sample_with_replacement=sample_with_replacement, weight_fac=weight_fac, logistic=args.logistic, n_tuning=0, one_per_group=False, rootdir=rootdir)

    # len(K_list) * len(signals) * len(n0_list) * len(s_list) * len(weight_facs)

    # i = 0
    # for nK, seed in itertools.product(K_list, np.arange(100)):
    #     if i == args.jobid:
    #         n1 = 8000 // (nK - 1)
    #         main(seed, args.w_replace, nK, n0, n1, args.p, s, signal_fac, weight_fac, args.logistic, 1000, root_dir=args.root_dir, aggregation_rule='majority')
    #     i += 1


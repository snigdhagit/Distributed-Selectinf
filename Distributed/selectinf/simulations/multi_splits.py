import numpy as np
import pandas as pd
import time
import itertools
from scipy.stats import norm
from collections import namedtuple
import os
from ..distributed_lasso import multisplit_lasso as L
from ..Tests.instance import gaussian_instance
from .metrics import get_metrics
import argparse
import pickle
from ..R_multicarve import *
import warnings
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.rinterface import RRuntimeWarning

# root = '/Users/sifanliu/Dropbox/Projects/Selective Inference/'
root = '/home/users/liusf19/'
robjects.r['source'](root + 'Multicarving/instance.R')
warnings.filterwarnings("ignore", category=RRuntimeWarning)

accuracy_metrics = namedtuple('Metrics', ['TP', 'FP', 'TN', 'FN', 'DOR'])
def get_accuracy(actual, predicted):
    TP = np.sum(actual * predicted)
    FP = np.sum((~actual) * predicted)
    TN = np.sum((~actual) * (~predicted))
    FN = np.sum(actual * (~predicted))
    DOR = (TP * TN) / (FP * FN)
    return accuracy_metrics(TP, FP, TN, FN, DOR)

def instance_R(seed, n, p, s, sigma, rho, signal):
    robjects.globalenv['seed'] = seed
    robjects.globalenv['n'] = n
    robjects.globalenv['p'] = p
    robjects.globalenv['s'] = s
    robjects.globalenv['sigma'] = sigma
    robjects.globalenv['rho'] = rho
    robjects.globalenv['signal'] = signal
    robjects.r('''
                inst = gaussian_instance(seed, n, p, s, sigma, rho, signal)
                X = inst[[1]]
                Y = inst[[2]]
                beta = inst[[3]]
            ''')
    return robjects.globalenv['X'], robjects.globalenv['Y'], robjects.globalenv['beta']

def run(seedn,
        n=1000,
        p=500,
        signal_fac=1.2,
        s=5,
        sigma=3,
        rho=0.9,
        nK=3,  # total number of machines, including holdout set
        B=10,
        proportion=None,
        level=0.9,
        sample_with_replacement=True,
        targets='selected',
        bonferroni_correct=False,
        n_tune=1000,
        gamma_min=0.05,
        gamma=None):
    np.random.seed(seedn)
    inst = gaussian_instance
    coverages = {}  # cover the target beta in selected/saturated view
    lengths = {}
    metrics = {}

    signal = np.sqrt(signal_fac * 2 * np.log(p))
    X, Y, beta = inst(n=n+n_tune,
                        p=p,
                        signal=signal,
                        s=s,
                        equicorrelated=False,
                        rho=rho,
                        sigma=sigma,
                        random_signs=True)[:3]
    # X, Y, beta = instance_R(seedn, n, p, s, sigma, rho, signal)
    if n_tune > 0:
        X_tune = X[n:]
        Y_tune = Y[n:]
        X = X[:n]
        Y = Y[:n]
    n, p = X.shape

    sigma_ = np.std(Y)
    
    
    methods = ['dist_carving', 'splitting', 'naive', 'multicarve']
    pvalues = {method: np.ones([B, p]) for method in methods}
    times = {method: 0. for method in methods}

    for b in range(B):  # B replicates
        ind = np.random.choice(n, int(n * proportion), replace=False)
        perturb = np.zeros([n, 1], dtype=bool)
        perturb[ind, :] = True
        # tune lambda
        weight_facs = np.linspace(0.1, 3, 10)
        min_mse = np.Inf
        best_weight_fac = weight_facs[0]
        for weight_fac in weight_facs:
            feature_weights_ = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ * weight_fac for i in range(1)}
            selector_ = L.gaussian(X,
                                    Y,
                                    feature_weights_,
                                    proportion,
                                    estimate_dispersion=False,
                                    sample_with_replacement=True)
            signs_ = selector_.fit(perturb=perturb)
            
            mse = np.linalg.norm(Y_tune - X_tune @ selector_._beta_full)**2
            # print(weight_fac, np.sum(signs_), mse)
            if mse < min_mse:
                min_mse = mse
                best_weight_fac = weight_fac
                selector = selector_
                signs = signs_

        print('best weight fac', best_weight_fac)
        signs = selector.fit()
        nonzero = signs != 0
        # screening = sum(true_signal * nonzero) == sum(true_signal)
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))  

            selector.setup_inference(dispersion=sigma) # dispersion=sigma

            target_spec = selector.selected_targets()
            
            t0 = time.time()
            result = selector.inference(target_spec,
                                        level=level)
            times['dist_carving'] += time.time() - t0

            pvalues['dist_carving'][b, nonzero] = result['pvalue']
            if bonferroni_correct:
                pvalues['dist_carving'][b, nonzero] = np.minimum(1, result['pvalue'] * np.sum(nonzero))
            # intervals = np.asarray(result[['lower_confidence',
            #                             'upper_confidence']])

            # coverage = (intervals[:, 0] < beta_target) * (intervals[:, 1] > beta_target)

            # length = intervals[:, 1] - intervals[:, 0]

            # print("check coverage + lengths ", np.mean(coverage), np.mean(length))
            # coverages['dist_carving'] = coverage
            # lengths['dist_carving'] = length

            # accept = (intervals[:, 0] < 0) * (intervals[:, 1] > 0)
            # metrics['dist_carving'] = get_metrics(beta[nonzero], accept)

            # splitting
            holdout_idx = np.sum(selector._selection_idx, 1) == 0
            X_holdout = X[holdout_idx][:, signs]
            Y_holdout = Y[holdout_idx]
            X_cov = np.linalg.inv(X_holdout.T @ X_holdout)
            splitting_estimator = X_cov @ X_holdout.T @ Y_holdout
            sds = np.sqrt(np.diag(X_cov)) * sigma
            pval = 2 * norm.cdf(-abs(splitting_estimator) / sds)
            pvalues['splitting'][b, nonzero] = pval
            if bonferroni_correct:
                pvalues['splitting'][b, nonzero] = np.minimum(1, pval * np.sum(nonzero))
            # coverages['splitting'] = abs(splitting_estimator - beta_target) < sds * norm.ppf(.5 + level / 2)
            # lengths['splitting'] = sds * 2 * norm.ppf(.5 + level / 2)

            # accept = abs(splitting_estimator) < sds * norm.ppf(.5 + level / 2)
            # metrics['splitting'] = get_metrics(beta[nonzero], accept)

            # naive intervals
            X_full_cov = np.linalg.inv(X[:, signs].T @ X[:, signs])
            full_sds = np.sqrt(np.diag(X_full_cov)) * sigma
            beta_ols = X_full_cov @ X[:, signs].T @ Y
            pval = 2 * norm.cdf(-abs(beta_ols) / full_sds)
            pvalues['naive'][b, nonzero] = pval
            if bonferroni_correct:
                pvalues['naive'][b, nonzero] = np.minimum(1, pval * np.sum(nonzero))
            # coverages['naive'] = abs(beta_ols - beta_target) < full_sds * norm.ppf(.5 + level / 2)
            # lengths['naive'] = full_sds * 2 * norm.ppf(.5 + level / 2)

            # accept = abs(beta_ols) < full_sds * norm.ppf(.5 + level / 2)
            # metrics['naive'] = get_metrics(beta[nonzero], accept)

            # multicarving
            print('multicarving')
            ind = np.where(np.sum(selector._selection_idx, 1) > 0)[0] + 1
            lbd = np.sqrt(2 * np.log(p)) * sigma_ * proportion * best_weight_fac
            beta_lasso = selector.beta_lasso.reshape(-1)
            pvalues['multicarve'][b, nonzero], mc_time = carve_lasso(X, Y, ind, beta_lasso, 1e-12, lbd, sigma, bonferroni_correct, level)
            times['multicarve'] += mc_time

    def aggregate_pvalues(pvalues, gamma_min=.05, gamma=None):
        if gamma is not None:
            print('gamma', gamma)
            q = np.quantile(pvalues / gamma, gamma, axis=0, interpolation='lower')
            return np.minimum(np.ones(p), q)

        gamma_list = np.arange(gamma_min, 1+1/B, 1/B)
        gamma_list = gamma_list[gamma_list <= 1.]
        agg_pvalues = np.ones(p)
        for gamma in gamma_list:
            q = np.quantile(pvalues, gamma, axis=0, interpolation='lower') / gamma
            agg_pvalues = np.minimum(agg_pvalues, q)
        return np.minimum(1, (1 - np.log(gamma_min)) * agg_pvalues)
    metrics = {}

    for method, pvals in pvalues.items():
        agg_pval = aggregate_pvalues(pvals, gamma_min=gamma_min, gamma=gamma)
        reject = agg_pval < (1 - level)
        metrics[method] = get_accuracy(beta != 0, reject)
        print(method, 'reject', np.sum(reject))
    return metrics, times

def main(proportion, seed, n, p, s, signal_fac, gamma_min=0.05, gamma=None, root_dir=''):
    nsim = 1
    print_every = 1
    methods = ['dist_carving', 'splitting', 'naive', 'multicarve']
    tp_ = {t: [] for t in methods}
    fp_ = {t: [] for t in methods}
    tn_ = {t: [] for t in methods}
    fn_ = {t: [] for t in methods}
    time_ = {t: [] for t in methods}
    dor_ = {t: [] for t in methods}

    # fcr_ = {t: [] for t in methods}
    # if sample_with_replacement:
    #     pass
    #     # proportion = .8
    #     # n = 1000
    # else:
    #     # proportion = np.ones(3) / 4
    #     n0 = 200
    #     n1 = 200
    #     n = (nK - 1) * n1 + n0
    #     proportion = np.ones(nK - 1) * (n1 / n)
    B = 5
    for i in range(1):
        metrics, times = run(seedn=seed, n=n, p=p, nK=2, sigma=2., signal_fac=signal_fac, s=s, B=B, proportion=proportion, sample_with_replacement=True, bonferroni_correct=False, gamma_min=gamma_min, gamma=gamma)
        # [dor_[key].append(metrics[key].DOR) for key in methods]
        # [tp_[key].append(metrics[key].TP) for key in methods]
        # [fp_[key].append(metrics[key].FP) for key in methods]
        # [tn_[key].append(metrics[key].TN) for key in methods]
        # [fn_[key].append(metrics[key][key].FN) for key in methods]

        # [time_[key].append(times[key]) for key in methods]


        # if (i + 1) % print_every == 0 or i == nsim - 1:
        df = pd.DataFrame(columns=methods, index=['DOR', 'time', 'TP', 'FP', 'TN', 'FN'])
        for key in methods:
            df[key] = [metrics[key].DOR, times[key], metrics[key].TP, metrics[key].FP, metrics[key].TN, metrics[key].FN]
        # print("========= Progress {:.1f}% =========".format(100 * (i + 1) / nsim))
        # print(pd.DataFrame(df))
        
        if root_dir == '':
            root_dir = 'Distributed/selectinf/simulations/results'

        os.makedirs(root_dir, exist_ok=True)
        if gamma is not None:
            filename = os.path.join(root_dir, f'multi_splits_prop_{proportion}_n_{n}_B_{B}_p_{p}_s_{s}_signal_{signal_fac}_gamma_{gamma}_seed_{seed}.csv')   
        else:
            filename = os.path.join(root_dir, f'multi_splits_prop_{proportion}_n_{n}_B_{B}_p_{p}_s_{s}_signal_{signal_fac}_gammamin_{gamma_min}_seed_{seed}.csv')   
        df.to_csv(filename)
        # results = {'precision': precision_, 'recall': recall_, 'f1': f1_, 'time': time_, 'dor': dor_}
        # results = {'metrics': metrics, 'times': times}
        # print(results)
        # print("saving to", filename)
        # with open(filename, 'wb') as f:
        #     pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--jobid', default=0, type=int)
    parser.add_argument('--root_dir', default='', type=str)
    args = parser.parse_args()


    props = [.5, .6, .7, .8, .9]
    gamma = args.gamma
    i = 0
    for proportion, seed in itertools.product(props, np.arange(200)):
        if i == args.jobid:
            print(proportion, seed)
            main(proportion=proportion, seed=seed, n=100, p=200, s=20, signal_fac=2., gamma_min=0.05, gamma=None, root_dir=args.root_dir)
            break
        i += 1
    
    

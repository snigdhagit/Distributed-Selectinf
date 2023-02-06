import numpy as np
import pandas as pd

from ..distributed_lasso import multisplit_lasso as L

from .instance import (gaussian_instance,
                       gaussian_grouped_instance,
                       logistic_instance,
                       logistic_grouped_instance)

import regreg.api as rr
from ..Utils.base import restricted_estimator

def aggregate_shavegroups(_overall):

    group_size = 5
    ngroup = 20

    groups = np.arange(ngroup).repeat(group_size)

    overall = _overall > 0
    new_overall = np.zeros(overall.shape[0], np.bool)
    for g in sorted(np.unique(groups)):

        srt = group_size * g
        stp = group_size * (g + 1)
        if overall[srt:stp].sum() > 0:
            rep = np.random.choice(group_size, 1, replace=False)
            new_overall[rep + srt] = True

    return new_overall

def aggregate_majority(_overall, threshold=1):

    new_overall = _overall > threshold
    #print("check ", _overall)
    return new_overall

def solve_target_restricted(linpred, X, active):

    def pi(x):
        return 1 / (1 + np.exp(-x))

    Y_mean = pi(linpred)
    n = X.shape[0]

    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))

    _beta_unpenalized = restricted_estimator(loglike,
                                             active)
    return _beta_unpenalized

def test(seedn,
         n=1000,
         p=500,
         signal_fac=0.1,
         s=5,
         sigma=1.,
         rho=0.9,
         Gaussian= True,
         targets='selected'):
    np.random.seed(seedn)

    while True:  # run until we get some selection

        signal = np.sqrt(signal_fac * 2 * np.log(p))

        if Gaussian is True:
            inst = gaussian_instance

            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=False,
                              rho=rho,
                              sigma=sigma,
                              random_signs=True)[:3]

            n_tuning = 1000

            # X, Y, beta= gaussian_grouped_instance(n=n+ n_tuning,
            #                                       p=p,
            #                                       s=s,
            #                                       sigma=sigma,
            #                                       rho=rho,
            #                                       signal=signal,
            #                                       random_signs=True,
            #                                       equicorrelated=True,
            #                                       n_group=20)[:3]

            if n_tuning > 0:
                X_tune = X[n:]  # used for tuning lambda
                Y_tune = Y[n:]
                X = X[:n]
                Y = Y[:n]
                n, p = X.shape

                sigma_ = np.std(Y)

                weight_facs = np.linspace(0.5, 2., 50)
                min_mse = np.Inf
                best_weight_fac = weight_facs[0]
                nK = 5
                proportion = np.ones(nK - 1)
                proportion /= nK
                for weight_fac in weight_facs:

                    feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ * weight_fac for i in
                                       range(nK - 1)}

                    selector_ = L.gaussian(X,
                                           Y,
                                           feature_weights,
                                           proportion,
                                           estimate_dispersion=True,
                                           sample_with_replacement=False)

                    signs_ = selector_.fit(aggregate=aggregate_shavegroups)
                    mse = np.linalg.norm(Y_tune - X_tune @ selector_._beta_full) ** 2
                    if mse < min_mse:
                        min_mse = mse
                        best_weight_fac = weight_fac
                        selector = selector_
                        signs = signs_
                print(best_weight_fac)

        else:
            # inst = logistic_instance
            #
            # X, Y, beta = inst(n=n,
            #                   p=p,
            #                   signal=signal,
            #                   s=s,
            #                   equicorrelated=False,
            #                   rho=rho,
            #                   random_signs=True)[:3]

            n_tuning = 1000
            inst = logistic_grouped_instance

            X, Y, beta = inst(n=n+n_tuning,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=True,
                              n_group=20)[:3]

            if n_tuning > 0:
                X_tune = X[n:]  # used for tuning lambda
                Y_tune = Y[n:]
                X = X[:n]
                Y = Y[:n]

                nK = 5
                proportion = np.ones(nK - 1)
                proportion /= nK

                linpred = X.dot(beta)
                sigma_ = 1
                feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) for i in range(nK - 1)}
                selector = L.logistic(X, Y, feature_weights, proportion)

                weight_facs = np.linspace(0.1, 1., num=70)
                max_loglik = -np.Inf
                best_weight_fac = weight_facs[0]

                for weight_fac in weight_facs:
                    feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * weight_fac for i in
                                       range(nK - 1)}
                    selector_ = L.logistic(X, Y, feature_weights, proportion)
                    signs_ = selector_.fit(aggregate=aggregate_shavegroups)
                    pi = X_tune @ selector_._beta_full
                    pi = 1 / (1 + np.exp(-pi))
                    loglik = np.sum(Y_tune * np.log(pi) + (1 - Y_tune) * np.log(1 - pi))
                    if loglik > max_loglik:
                        max_loglik = loglik
                        best_weight_fac = weight_fac
                        selector = selector_
                        signs = signs_
                print(best_weight_fac)

        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            if Gaussian is True:
                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

                selector.setup_inference(dispersion=None)

            else:
                beta_target = solve_target_restricted(linpred, X, selector.overall)

                selector.setup_inference(dispersion=1.)

            target_spec = selector.selected_targets()

            result = selector.inference(target_spec,
                                        level=0.90)

            intervals = np.asarray(result[['lower_confidence',
                                           'upper_confidence']])

            coverage = (intervals[:, 0] < beta_target) * (intervals[:, 1] > beta_target)

            length = intervals[:, 1] - intervals[:, 0]

            #print("check coverage + lengths ", np.mean(coverage), np.mean(length))

            return coverage, length


def main(nsim=500):
    cover_ = []
    len_ = []

    for i in range(nsim):
        coverage, length = test(seedn=i,
                                n=8000,
                                p=100,
                                Gaussian=True)

        cover_.extend(coverage)
        len_.extend(length)

        print("Coverage so far ", i, np.mean(cover_))
        print("Lengths so far ", i, np.mean(len_))

main(nsim=500)
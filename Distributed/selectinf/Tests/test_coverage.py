import numpy as np
import pandas as pd

from ..distributed_lasso import multisplit_lasso as L

from .instance import (gaussian_instance,
                       logistic_instance)

import regreg.api as rr
from ..Utils.base import restricted_estimator

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
         signal_fac=0.5,
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

            n, p = X.shape

            sigma_ = np.std(Y)

            nK = 20

            feature_weights = {i: 2* np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ for i in range(nK - 1)}
            proportion = np.ones(nK - 1)
            proportion /= nK

            selector = L.gaussian(X,
                                  Y,
                                  feature_weights,
                                  proportion,
                                  estimate_dispersion=True)

        else:
            inst = logistic_instance

            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=False,
                              rho=rho,
                              random_signs=True)[:3]

            linpred = X.dot(beta)

            nK = 20
            sigma_ = 1

            feature_weights = {i: 0.8* np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ for i in range(nK - 1)}
            proportion = np.ones(nK - 1)
            proportion /= nK

            selector = L.logistic(X,
                                  Y,
                                  feature_weights,
                                  proportion)


        signs = selector.fit()
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

            print("check coverage + lengths ", np.mean(coverage), np.mean(length))

            return coverage, length


def main(nsim=500):
    cover_ = []
    len_ = []

    for i in range(nsim):
        coverage, length = test(seedn=i,
                                n=10000,
                                p=100,
                                Gaussian= True)

        cover_.extend(coverage)
        len_.extend(length)

        print("Coverage so far ", i, np.mean(cover_))
        print("Lengths so far ", i, np.mean(len_))

main(nsim=500)
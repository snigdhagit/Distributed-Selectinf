import numpy as np
import pandas as pd

from ..distributed_lasso import multisplit_lasso as L

from .instance import gaussian_instance


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
        print("check column norms ", np.diag(X.T.dot(X)))

        sigma_ = np.std(Y)

        if n > (2 * p):
            full_dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            full_dispersion = sigma_ ** 2

        nK = 2

        feature_weights = {i: np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_ for i in range(nK - 1)}
        proportion = 1 / 2.

        selector = L.gaussian(X,
                              Y,
                              feature_weights,
                              proportion,
                              estimate_dispersion=True)

        signs = selector.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

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

            print("check coverage + lengths ", np.mean(coverage), np.mean(length))

            return coverage, length


def main(nsim=500):
    cover_ = []
    len_ = []

    for i in range(nsim):
        coverage, length = test(seedn=i,
                                n=3000,
                                p=500)

        cover_.extend(coverage)
        len_.extend(length)

        print("Coverage so far ", i, np.mean(cover_))
        print("Lengths so far ", i, np.mean(len_))

main(nsim=100)
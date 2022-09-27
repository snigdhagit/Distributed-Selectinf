from __future__ import print_function

import numpy as np
import regreg.api as rr
from scipy.linalg import block_diag

from .query import gaussian_query

from .Utils.base import (restricted_estimator,
                         _compute_hessian,
                         _pearsonX2)

from typing import NamedTuple

class TargetSpec(NamedTuple):
    observed_target: np.ndarray
    cov_target: np.ndarray
    regress_target_score: np.ndarray
    alternatives: list


class multisplit_lasso(gaussian_query):

    def __init__(self,
                 loglike,
                 feature_weights,
                 proportion,
                 nsamples,
                 nfeatures,
                 estimate_dispersion=True):

        self.loglike = loglike

        self.feature_weights = feature_weights

        self.proportion = proportion

        self.nsplit = len(feature_weights)

        self.n = nsamples

        self.p = nfeatures

        self.estimate_dispersion = estimate_dispersion

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        (initial_solns,
         initial_subgrads,
         active,
         active_signs) = self.solve_LASSOs(perturb=perturb,
                                           solve_args=solve_args)

        _overall = np.sum(active, axis=1)
        overall = _overall>0

        nactive = overall.sum()

        inactive = ~overall

        ordered_variables = {i: list(tuple(np.nonzero(active[:,i])[0])) for i in range(self.nsplit)}

        self.selection_variable = {i: {'sign': active_signs[:,i], 'variables': ordered_variables[i]} for i in range(self.nsplit)}

        self.observed_opt_states = {i: np.fabs((initial_solns[i,:])[active[:,i]]) for i in range(self.nsplit)}

        num_opt_var = np.array([self.observed_opt_states[i].shape[0] for i in range(self.nsplit)])

        _beta_unpenalized = restricted_estimator(self.loglike,
                                                 overall,
                                                 solve_args=solve_args)

        beta_bar = np.zeros(self.p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        unpenalized = np.zeros(self.p, dtype=bool)

        _hessian, _hessian_active, _hessian_unpen = _compute_hessian(self.loglike,
                                                                     beta_bar,
                                                                     overall,
                                                                     unpenalized)

        #print("check shapes ", _hessian_active.shape, nactive, num_opt_var)

        _score_linear_term = np.zeros((self.p, nactive))

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])

        # set the observed score (data dependent) state

        # observed_score_state is
        # \nabla \ell(\bar{\beta}_E) - Q(\bar{\beta}_E) \bar{\beta}_E
        # in linear regression this is _ALWAYS_ -X^TY
        #
        # should be asymptotically equivalent to
        # \nabla \ell(\beta^*) - Q(\beta^*)\beta^*

        _observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        _observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        self.observed_score_state = np.tile(_observed_score_state, self.nsplit)

        opt_linear = []

        opt_linear_dict = {}

        X, Y = self.loglike.data

        observed_subgrad = []

        signed_XE = {}

        _observed_opt_state = []

        for i in range(self.nsplit):

            _observed_opt_state.append(np.fabs((initial_solns[i,:])[active[:,i]]))

            _opt_linear = np.zeros((self.p, num_opt_var[i]))

            scaling_slice = slice(0, active[:, i].sum())

            if np.sum(active[:, i]) == 0:
                _opt_hessian = 0
            else:
                hess_active = X.T.dot(X[:, active[:, i]])
                _opt_hessian = hess_active * (active_signs[:, i])[None, active[:, i]]

            _opt_linear[:, scaling_slice] = _opt_hessian
            opt_linear_dict[i]= _opt_linear
            opt_linear = block_diag(opt_linear, _opt_linear)

            observed_subgrad.append(initial_subgrads[i,:])

            signed_XE[i] = X[:, active[:, i]].dot(np.diag(active_signs[:, i][active[:, i]]))

        opt_linear = opt_linear[1:, :]
        observed_subgrad = np.ravel(np.asarray(observed_subgrad))
        self.observed_opt_state = [item for arr in _observed_opt_state for item in arr]

        ####check KKT conditions
        for i in range(self.nsplit):

            omega = -(self.randomized_loss_list[i]).smooth_objective(initial_solns[i, :],'grad') \
                    + self.loglike.smooth_objective(initial_solns[i, :],'grad')
            # print("check K.K.T. Mapping ", i, np.allclose(omega, -X.T.dot(Y) + (opt_linear_dict[i]).dot(self.observed_opt_states[i])+ initial_subgrads[i,:]))

        # now make the constraints and implied gaussian

        self._setup = True
        A_scaling = -np.identity(num_opt_var.sum())
        b_scaling = np.zeros(num_opt_var.sum())

        #### to be fixed -- set the cov_score here without dispersion

        self._unscaled_cov_score = _hessian

        self._setup_sampler_data = (A_scaling,
                                    b_scaling,
                                    opt_linear,
                                    observed_subgrad)

        if self.estimate_dispersion:

            X, y = self.loglike.data
            n, p = X.shape

            dispersion = 2 * (self.loglike.smooth_objective(self._beta_full,
                                                            'func') /(n - nactive))

            self.dispersion_ = dispersion

        self.df_fit = nactive

        self.num_opt_var = num_opt_var

        self.signed_XE = signed_XE

        self.opt_linear_dict = opt_linear_dict

        self._hessian = _hessian

        self._hessian_active = _hessian_active

        self.overall = overall

        return overall

    def setup_inference(self,
                        dispersion):

        if self.df_fit > 0:

            if dispersion is None:
                self._setup_sampler(*self._setup_sampler_data,
                                    dispersion=self.dispersion_)

            else:
                self._setup_sampler(*self._setup_sampler_data,
                                    dispersion=dispersion)

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        pi_s = self.proportion

        K = self.nsplit

        a = (pi_s ** 2)/(1-pi_s*K)

        diag_ = (pi_s + a) * 1./dispersion

        off_diag_ = a * 1./dispersion

        diag_precision_ = np.array([])

        for i in range(K):

            ordered_vars = (self.selection_variable[i])['variables']

            D = (self.opt_linear_dict[i])[ordered_vars] * diag_

            signs = (self.selection_variable[i])['sign'][ordered_vars]
            signs[np.isnan(signs)] = 1

            D *= signs[:, None]

            diag_precision_ = block_diag(diag_precision_, D)

        diag_precision_ = diag_precision_[1:, :]

        Cum_nactive = np.cumsum(self.num_opt_var)

        for r in range(K):

            N_ = np.zeros((self.num_opt_var[r],Cum_nactive[r]))

            upper_tri_precision= N_

            if r<K-1:
                cols = np.arange(K-r-1) + (r + 1)
                for c in cols:
                    upper_tri_precision = np.hstack((upper_tri_precision, self.signed_XE[r].T.dot(self.signed_XE[c])))

            upper_tri_precision *= off_diag_

            if r == 0:
                off_diag_precision = upper_tri_precision
            else:
                off_diag_precision = np.vstack((off_diag_precision, upper_tri_precision))

        off_diag_precision_ = off_diag_precision + off_diag_precision.T

        cond_precision = off_diag_precision_ + diag_precision_  # \Gamma^{-1}

        cond_cov = np.linalg.inv(cond_precision)  # \Gamma

        B_ = []

        for i in range(K):

            ordered_vars = (self.selection_variable[i])['variables']

            signs = (self.selection_variable[i])['sign'][ordered_vars]
            signs[np.isnan(signs)] = 1

            R_ = np.zeros((len(ordered_vars), self.p))

            R_[:, ordered_vars] = np.identity(len(ordered_vars)) * signs[None, :]
            #print("Assignment ", R_)
            if i == 0:
                regress_opt_= R_
            else:
                regress_opt_ = np.vstack((regress_opt_, R_))

            B_ = block_diag(B_, pi_s * R_)

        B_ = B_[1:, :]

        regress_opt = -(cond_cov.dot(B_ + np.tile(regress_opt_, K) * a)) * 1./dispersion

        #print("Regress Opt ", cond_cov/dispersion, B_ + np.tile(regress_opt_, K) * a, regress_opt)

        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        #print("Cond mean ", cond_mean)

        I_ = np.tile(np.identity(self.p), [K,K]) * a + np.kron(np.eye(K), np.identity(self.p)) * pi_s

        prod_score_prec_unnorm = I_ * 1./dispersion

        U = (np.identity(K) * 1./pi_s) - np.ones((K,K))

        cov_rand = np.kron(U, self._unscaled_cov_score) * dispersion

        self.cov_rand = cov_rand

        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                M1,
                M2,
                M3)

    def solve_LASSOs(self,
                     perturb=None,
                     solve_args={'tol': 1.e-12, 'min_its': 50}):

        if perturb is not None:
            self._selection_idx = perturb
        if not hasattr(self, "_selection_idx"):

            if self.nsplit == 1:
                pi_s = self.proportion
                self._selection_idx = np.zeros((self.n,1), np.bool)
                (self._selection_idx[:, 0])[:int(pi_s * self.n)] = True
                np.random.shuffle(self._selection_idx[:,0])

            else:
                _selection_idx = np.zeros((self.n, self.nsplit), np.bool)

                nsize = int(self.proportion * self.n)

                for i in range(self.nsplit):
                    _start = i * nsize
                    _stop = (i + 1) * nsize
                    (_selection_idx[:, i])[_start: _stop] = True
                    self._selection_idx = np.random.permutation(_selection_idx)


        feature_weights_list = [rr.weighted_l1norm(self.feature_weights[i], lagrange=1.) for i in range(self.nsplit)]

        quad = rr.identity_quadratic(0,0,0,0)

        inv_frac = 1 / self.proportion

        randomized_loss_list = [self.loglike.subsample(self._selection_idx[:,i]) for i in range(self.nsplit)]

        for i in range(self.nsplit):
            (randomized_loss_list[i]).coef *= inv_frac

        problem_list = [rr.simple_problem(randomized_loss_list[i], feature_weights_list[i]) for i in range(self.nsplit)]

        initial_solns = np.asarray([problem_list[i].solve(quad, **solve_args) for i in range(self.nsplit)])

        initial_subgrads = np.asarray([-((randomized_loss_list[i]).smooth_objective(initial_solns[i],'grad')
                                       + quad.objective(initial_solns[i], 'grad')) for i in range(self.nsplit)])

        active_signs = np.zeros((self.p, self.nsplit))
        active = np.zeros((self.p, self.nsplit), np.bool)
        #print("check shape ", initial_solns.shape, initial_subgrads.shape)

        for i in range(self.nsplit):
            active_signs[:, i] = np.sign(initial_solns[i, :])
            active[:, i] = active_signs[:, i] != 0

        self.randomized_loss_list = randomized_loss_list
        return initial_solns, initial_subgrads, active, active_signs

    def selected_targets(self,
                         sign_info={},
                         solve_args={'tol': 1.e-12, 'min_its': 100}):

        observed_target = restricted_estimator(self.loglike,
                                               self.overall,
                                               solve_args=solve_args)

        Qfeat = self._hessian_active[self.overall]

        cov_target = np.linalg.inv(Qfeat)

        alternatives = ['twosided'] * self.overall.sum()
        features_idx = np.arange(self.p)[self.overall]

        for i in range(len(alternatives)):
            if features_idx[i] in sign_info.keys():
                alternatives[i] = sign_info[features_idx[i]]

        _regress_target_score = np.zeros((cov_target.shape[0], self.p))
        _regress_target_score[:, self.overall] = cov_target
        regress_target_score = np.tile(_regress_target_score, self.nsplit)

        #print("check shape of regress_target_score ", regress_target_score.shape)
        # print("Dispersion check", self.dispersion_)

        return TargetSpec(observed_target,
                          cov_target * self.dispersion_,
                          regress_target_score,
                          alternatives)

    @staticmethod
    def gaussian(X,
                 Y,
                 feature_weights,
                 proportion,
                 sigma=1.,
                 quadratic=None,
                 estimate_dispersion=True):

        loglike = rr.glm.gaussian(X,
                                  Y,
                                  coef=1. / sigma ** 2,
                                  quadratic=quadratic)

        nsamples = X.shape[0]
        nfeatures = X.shape[1]

        return multisplit_lasso(loglike,
                                feature_weights,
                                proportion,
                                nsamples,
                                nfeatures,
                                estimate_dispersion=estimate_dispersion)



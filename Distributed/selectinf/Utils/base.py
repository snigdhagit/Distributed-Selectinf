import numpy as np
from collections import namedtuple
import regreg.api as rr
import regreg.affine as ra


def restricted_estimator(loss, active, solve_args={'min_its':50, 'tol':1.e-10}):
    """
    Fit a restricted model using only columns `active`.

    Parameters
    ----------

    Mest_loss : objective function
        A GLM loss.

    active : ndarray
        Which columns to use.

    solve_args : dict
        Passed to `solve`.

    Returns
    -------

    soln : ndarray
        Solution to restricted problem.

    """
    X, Y = loss.data

    if not loss._is_transform and hasattr(loss, 'saturated_loss'): # M_est is a glm
        X_restricted = X[:,active]
        loss_restricted = rr.affine_smooth(loss.saturated_loss, X_restricted)
    else:
        I_restricted = ra.selector(active, ra.astransform(X).input_shape[0], ra.identity((active.sum(),)))
        loss_restricted = rr.affine_smooth(loss, I_restricted.T)
    beta_E = loss_restricted.solve(**solve_args)
    
    return beta_E


def logistic_target_restricted(linpred, X, active):

    def pi(x):
        return 1 / (1 + np.exp(-x))

    Y_mean = pi(linpred)
    n = X.shape[0]

    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))

    _beta_unpenalized = restricted_estimator(loglike,
                                             active)
    return _beta_unpenalized


def _compute_hessian(loglike,
                     beta_bar,
                     *bool_indices):

    X, y = loglike.data
    linpred = X.dot(beta_bar)
    n = linpred.shape[0]

    if hasattr(loglike.saturated_loss, "hessian"): # a GLM -- all we need is W
        W = loglike.saturated_loss.hessian(linpred)
        parts = [np.dot(X.T, X[:, bool_idx] * W[:, None]) for bool_idx in bool_indices]
        _hessian = np.dot(X.T, X * W[:, None]) # CAREFUL -- this will be big
    elif hasattr(loglike.saturated_loss, "hessian_mult"):
        parts = []
        for bool_idx in bool_indices:
            _right = np.zeros((n, bool_idx.sum()))
            for i, j in enumerate(np.nonzero(bool_idx)[0]):
                _right[:,i] = loglike.saturated_loss.hessian_mult(linpred, 
                                                                       X[:,j], 
                                                                       case_weights=loglike.saturated_loss.case_weights)
            parts.append(X.T.dot(_right))
        _hessian = np.zeros_like(X)
        for i in range(X.shape[1]):
            _hessian[:,i] = loglike.saturated_loss.hessian_mult(linpred, 
                                                                     X[:,i], 
                                                                     case_weights=loglike.saturated_loss.case_weights)
        _hessian = X.T.dot(_hessian)
    else:
        raise ValueError('saturated_loss has no hessian or hessian_mult method')

    if bool_indices:
        return (_hessian,) + tuple(parts)
    else:
        return _hessian


def _pearsonX2(y,
               linpred,
               loglike,
               df_fit):

    W = loglike.saturated_loss.hessian(linpred)
    n = y.shape[0]
    resid = y - loglike.saturated_loss.mean_function(linpred)
    return (resid ** 2 / W).sum() / (n - df_fit)


def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):

    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(U1)))
    U5 = U1.T.dot(QS.M1.dot(QS.opt_linear))

    return U1, U2, U3, U4, U5


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


Metrics = namedtuple('Metrics', ['TP', 'TN', 'FP', 'FN', 'F1', 'DOR'])
def get_metrics(beta_target, reject):
    accept = ~reject
    tp = np.sum(reject * (beta_target != 0))
    tn = np.sum(accept * (beta_target == 0))
    fp = np.sum(reject * (beta_target == 0))
    fn = np.sum(accept * (beta_target != 0))
    f1_score = 2 * tp / (2 * tp + fp + fn)
    dor = (tp * tn) / (fp * fn)
    return Metrics(tp, tn, fp, fn, f1_score, dor)

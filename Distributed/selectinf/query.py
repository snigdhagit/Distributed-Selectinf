from typing import NamedTuple
import numpy as np, pandas as pd

from .selective_MLE import mle_inference

class QuerySpec(NamedTuple):

    # law of o|S,u

    cond_mean : np.ndarray
    cond_cov : np.ndarray

    # how S enters into E[o|S,u]

    opt_linear : np.ndarray

    # constraints

    linear_part : np.ndarray
    offset : np.ndarray

    # score / randomization relationship

    M1 : np.ndarray
    M2 : np.ndarray
    M3 : np.ndarray

    # observed values

    observed_opt_state : np.ndarray
    observed_score_state : np.ndarray
    observed_subgrad : np.ndarray
    observed_soln : np.ndarray
    observed_score : np.ndarray
    
class gaussian_query(object):
    r"""
    This class is the base of randomized selective inference
    based on convex programs.
    The main mechanism is to take an initial penalized program
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B)
    and add a randomization and small ridge term yielding
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B) -
        \langle \omega, B \rangle + \frac{\epsilon}{2} \|B\|^2_2
    """

    def __init__(self, randomization, perturb=None):

        """
        Parameters
        ----------
        randomization : `selection.randomized.randomization.randomization`
            Instance of a randomization scheme.
            Describes the law of $\omega$.
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """
        self.randomization = randomization
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    @property
    def specification(self):
        return QuerySpec(cond_mean=self.cond_mean,
                         cond_cov=self.cond_cov,
                         opt_linear=self.opt_linear,
                         linear_part=self.linear_part,
                         offset=self.offset,
                         M1=self.M1,
                         M2=self.M2,
                         M3=self.M3,
                         observed_opt_state=self.observed_opt_state,
                         observed_score_state=self.observed_score_state,
                         observed_subgrad=self.observed_subgrad,
                         observed_soln=self.observed_opt_state,
                         observed_score=self.observed_score_state + self.observed_subgrad)
   

    # Methods reused by subclasses

    def randomize(self, perturb=None):

        """
        The actual randomization step.
        Parameters
        ----------
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """

        if not self._randomized:
            (self.randomized_loss,
             self._initial_omega) = self.randomization.randomize(self.loss,
                                                                 self.epsilon,
                                                                 perturb=perturb)
        self._randomized = True

    def get_sampler(self):
        if hasattr(self, "_sampler"):
            return self._sampler

    def set_sampler(self, sampler):
        self._sampler = sampler

    sampler = property(get_sampler, set_sampler, doc='Sampler of optimization (augmented) variables.')

    def fit(self, perturb=None):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

    # Private methods

    def _setup_sampler(self,
                       linear_part,
                       offset,
                       opt_linear,
                       observed_subgrad,
                       dispersion=1):

        A, b = linear_part, offset

        if not np.all(A.dot(self.observed_opt_state) - b <= 0):
            raise ValueError('constraints not satisfied')

        (cond_mean,
         cond_cov,
         cond_precision,
         M1,
         M2,
         M3) = self._setup_implied_gaussian(opt_linear,
                                            observed_subgrad,
                                            dispersion=dispersion)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov

        self.opt_linear = opt_linear
        self.observed_subgrad = observed_subgrad
        self.linear_part = linear_part
        self.offset = offset

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        cov_rand, prec = self.randomizer.cov_prec

        if np.asarray(prec).shape in [(), (0,)]:
            prod_score_prec_unnorm = self._unscaled_cov_score * prec
        else:
            prod_score_prec_unnorm = self._unscaled_cov_score.dot(prec)

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T) * prec
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T).dot(prec)

        # regress_opt is regression coefficient of opt onto score + u...

        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

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

    def inference(self,
                  target_spec,
                  level=0.90,
                  method_args={}):

        """
        Parameters
        ----------
        target_spec : TargetSpec
           Information needed to specify the target.
        method : str
           One of ['selective_MLE', 'approx', 'exact', 'posterior']
        level : float
           Confidence level or posterior quantiles.
        method_args : dict
           Dict of arguments to be optionally passed to the methods.

        Returns
        -------

        summary : pd.DataFrame
           Statistical summary for specified targets.
        """

        query_spec = self.specification

        G = mle_inference(query_spec,
                          target_spec,
                          **method_args)

        return G.solve_estimating_eqn(alternatives=target_spec.alternatives,
                                      level=level)[0]













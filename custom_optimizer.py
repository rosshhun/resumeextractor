import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score
import math
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class HyperbandSearchCV(BaseEstimator):
    def __init__(self, estimator, param_distributions, resource_param, min_iter=1, max_iter=81, eta=3, scoring=None,
                 n_jobs=-1, cv=5, random_state=None, verbose=0):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.resource_param = resource_param
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.eta = eta
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    def fit(self, X, y):
        def run_iteration(n, r):
            T = [self._get_random_config() for _ in range(n)]
            for i in range(int(math.log(self.max_iter / r, self.eta))):
                n_i = int(n * self.eta ** (-i))
                r_i = r * self.eta ** i

                logger.info(f"Iteration {i + 1}: Evaluating {n_i} configurations with {r_i} resources each")

                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._evaluate_config)(t, r_i, X, y) for t in T[:n_i]
                )

                losses, configs = zip(*results)

                top_indices = np.argsort(losses)[:int(n_i / self.eta)]
                T = [configs[i] for i in top_indices]

            return T[0]

        s_max = int(math.log(self.max_iter / self.min_iter, self.eta))
        B = (s_max + 1) * self.max_iter

        logger.info(f"Starting Hyperband optimization with s_max={s_max}, B={B}")

        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)

            logger.info(f"Bracket s={s}: n={n}, r={r}")

            best_config = run_iteration(n, r)
            score = self._evaluate_config(best_config, self.max_iter, X, y)[0]

            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = best_config
                self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_)
                self.best_estimator_.fit(X, y)

                logger.info(f"New best score: {self.best_score_:.4f}")
                logger.info(f"New best params: {self.best_params_}")

        logger.info("Hyperband optimization completed")
        return self

    def _get_random_config(self):
        sampler = ParameterSampler(self.param_distributions, n_iter=1, random_state=self.random_state)
        return next(iter(sampler))

    def _evaluate_config(self, config, resource, X, y):
        estimator = clone(self.estimator)
        estimator.set_params(**{**config, self.resource_param: int(resource)})
        scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring, n_jobs=1)
        return -np.mean(scores), config  # Negative because Hyperband minimizes

    def score(self, X, y):
        return self.best_estimator_.score(X, y)


class CustomInteger:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def rvs(self, random_state=None):
        return np.random.randint(self.low, self.high + 1)
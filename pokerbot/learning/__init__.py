"""Learning subpackage — the AI layer that detects HUMAN behaviors.

The headline components are bluff and tilt classifiers — these are the
"AI catches humans being human" parts. The other modules are supporting
infrastructure that the classifiers (or the policy) can use.

Module map (with course topics):
  - bluff_classifier.py  -> P(opponent is bluffing | features)
                            [Probabilistic Reasoning, MLE/optimization]
  - tilt_classifier.py   -> P(opponent is on tilt | recent-behavior features)
                            [Probabilistic Reasoning, MLE/optimization]
  - gmm.py               -> Discover player archetypes from stat vectors
                            [Expectation Maximization, Gaussian Mixture Models]
  - mle_strategy.py      -> Estimate opponent's mixed strategy from observed
                            actions, with Dirichlet prior
                            [Probabilistic Reasoning, MLE, Bayesian inference]
"""

from pokerbot.learning.mle_strategy import StrategyMLE
from pokerbot.learning.gmm import GaussianMixture, fit_gmm
from pokerbot.learning.bluff_classifier import BluffClassifier, FEATURES as BLUFF_FEATURES
from pokerbot.learning.tilt_classifier import TiltClassifier, TILT_FEATURES
from pokerbot.learning.validation import (
    auc,
    brier_score,
    expected_calibration_error,
    reliability_diagram,
    cross_validate,
    CrossValReport,
    ReliabilityCurve,
)

__all__ = [
    "StrategyMLE",
    "GaussianMixture",
    "fit_gmm",
    "BluffClassifier",
    "BLUFF_FEATURES",
    "TiltClassifier",
    "TILT_FEATURES",
    "auc",
    "brier_score",
    "expected_calibration_error",
    "reliability_diagram",
    "cross_validate",
    "CrossValReport",
    "ReliabilityCurve",
]

# AI / ML components — mapping to the course outline

This document points out where each AI topic from the course actually shows up
in the codebase, and what it does for the bot. The headline AI elements (per
the project goal) are **detection of human-element behaviors — bluffs and tilt**.
Everything else is supporting infrastructure that the detection layer uses.

| Course topic / week | Where it lives | What it does |
|---|---|---|
| **Probabilistic Reasoning** (Wks 5–6) | `pokerbot/opponent/stats.py`, `pokerbot/opponent/archetypes.py`, `pokerbot/learning/mle_strategy.py` | Bayesian posterior over opponent archetypes; Beta/Dirichlet priors with empirical-Bayes shrinkage; per-info-set posterior over opponent's mixed strategy |
| **MLE & Optimization** (Wk 7) | `pokerbot/learning/bluff_classifier.py`, `pokerbot/learning/tilt_classifier.py`, `pokerbot/learning/mle_strategy.py` | Logistic regression fit by gradient descent on the negative log-likelihood (the canonical MLE-by-optimization workflow); Dirichlet-MAP for opponent strategies |
| **EM** (Wk 8) | `pokerbot/learning/gmm.py` | Hand-written Expectation-Maximization loop (E-step computes responsibilities, M-step updates means/vars/weights). Used to discover player archetypes from a dataset of opponent stat vectors |
| **GMMs** (Wk 10) | `pokerbot/learning/gmm.py` | Diagonal-covariance Gaussian mixture model with K-means++ initialization. Test verifies it recovers known cluster means and that log-likelihood is monotone non-decreasing |
| **Reinforcement Learning** (Wks 11–12) | (intentionally omitted per scope) | The CFR/MCCFR self-play training loops (`pokerbot/solvers/`) play the same role as RL self-play: they iteratively improve a policy against itself. CFR is technically *online learning under imperfect information* — closely related to RL but with the imperfect-info structure built in. If we wanted standard Q-learning later, the game abstraction in `pokerbot/games/base.py` already exposes the API it needs |
| **Topic 1 / Text analysis** (Wk 13) | (n/a for this project) | — |
| **Transformers / LLMs / Agentic AI** (Wk 14) | (n/a for this project) | — |

---

## Where the AI actually decides things during play

The decision pipeline lives in `pokerbot/policy/human_aware.py`:

```
opponent observes new action
    ↓
StrategyMLE.observe(I, a)              [Probabilistic Reasoning, Bayesian update]
    ↓
when bot must act:
    ↓
BluffClassifier.predict_proba(features) → P(bluff)
    ↓                                    [MLE-fit logistic regression]
TiltClassifier.predict_proba(features) → P(tilt)
    ↓                                    [MLE-fit logistic regression]
adjust_strategy_for_bluff_signal(modeled, P(bluff))
    ↓
effective_lambda ← base_lambda * (1 + 0.5 * P(tilt))
    ↓
HybridPolicy = (1−λ) · GTO + λ · BR(modeled)   [the Libratus pattern]
    ↓
action sampled from HybridPolicy
```

The two classifier heads are exactly where the AI judges humans being human;
everything upstream and downstream is unexploitable game-theoretic poker.

## Test coverage that validates each component

- `tests/test_mle_strategy.py` — 6 tests (Dirichlet shrinkage, convergence, confidence growth)
- `tests/test_gmm.py` — 5 tests (LL monotone, recovers means, responsibilities valid, >95% clustering accuracy)
- `tests/test_bluff_classifier.py` — 4 tests (loss decreases, AUC > 0.85, expected feature importances)
- `tests/test_tilt_classifier.py` — 3 tests (>85% accuracy, loss decreases, recent-loss feature carries weight)
- `tests/test_opponent.py` — 10 tests on Bayesian archetype posterior + Beta shrinkage + heuristic tilt
- `tests/test_hybrid.py` — 5 tests (mixing math correct, hybrid beats GTO vs weak, doesn't lose vs Nash)

Plus the GTO foundation: `tests/test_cfr_convergence.py`, `tests/test_exploitability.py`,
`tests/test_leduc_convergence.py`, `tests/test_mccfr.py`.

## How to see it work

```bash
# 1. The CFR foundation: solver finds Nash on Kuhn
python -m scripts.train_kuhn 20000

# 2. The CFR foundation: exploitability falls on Leduc
python -m scripts.train_leduc 200

# 3. The hybrid (GTO + BR) layer: massive gains vs weak opps
python -m scripts.head_to_head

# 4. THE MAIN DEMO: bluff + tilt classifiers swing the bot's play
python -m scripts.demo_human_aware

# 5. Run the whole test suite
python -m unittest discover tests
```

# Paper Analysis — What Each Algorithm Actually Does

These notes drive the architecture of this bot. Each section pulls out the concrete idea we're going to implement.

---

## 1. Counterfactual Regret Minimization (CFR) — Zinkevich et al. 2007

**What it is:** An iterative self-play algorithm that converges to a Nash equilibrium in two-player zero-sum extensive-form games with imperfect information.

**Core mechanic:**
- For every information set `I` and every action `a`, track *cumulative counterfactual regret* `R^T(I, a)`.
- The current strategy at `I` is `regret-matching`: probabilities proportional to `max(0, R^T(I,a))`.
- After many iterations, the *average* strategy (not the current one) approaches Nash.

**Counterfactual** part: when computing regret at `I`, we *only* count utility weighted by the probability the opponent and chance reach `I` — i.e., we pretend our player wanted to reach `I`. This is what lets the regrets compose into a global Nash strategy.

**Why average strategy:** Regret-matching's *current* strategy oscillates. The time-average is what converges.

**Implementation footprint (vanilla CFR):**
- `cfr(history, reach_probs, player_to_train) -> utility`
- For each action: recurse with reach probs updated by current strategy.
- Update regret = (counterfactual_value(a) − counterfactual_value(strategy)).
- Update strategy_sum (for the average).

**Files in this repo:** `pokerbot/solvers/cfr.py`

---

## 2. Monte Carlo CFR (MCCFR) — Lanctot et al. 2009

**Why we need it:** Vanilla CFR traverses the entire game tree every iteration. For HU NLHE, that's >10^160 states. Infeasible.

**External sampling MCCFR:** On each iteration, *sample* chance outcomes and the opponent's actions; only enumerate the training player's actions exhaustively. Variance is higher per iteration but each iteration is cheap.

**Outcome sampling MCCFR:** Sample one full trajectory, weight regret updates by 1/sampling_probability. Even cheaper, higher variance.

**Files:** `pokerbot/solvers/mccfr.py` — we implement external sampling.

---

## 3. CFR+ — Tammelin 2014 (used in Cepheus, the HULHE-solving bot)

Two changes to vanilla CFR that empirically converge ~10× faster:
1. **Regret floor at 0.** Negative regrets are clamped each iteration instead of accumulating.
2. **Linear averaging.** Weight iteration `t`'s contribution to the average strategy by `t` (later iterations matter more).

Drop-in replacement once vanilla CFR works.

---

## 4. DeepStack — Moravčík et al. 2017

First bot to beat pros at HU NLHE. **Key idea: continual re-solving with a value network.**

- Don't store a precomputed strategy for the whole game (too big).
- At each decision, *re-solve* a depth-limited subgame in real time.
- Replace the leaves of the subgame with a **neural network** that predicts the counterfactual values of holding each hand at that node, given the range distribution.
- This is essentially "alpha-beta with a learned value function," but for imperfect-info games.

**Why this is hard:** in poker, the value of a node depends on the *range distribution* of both players, not just a single state. The net's input is a probability distribution over hands.

**What we'll borrow:** Depth-limited solving (later phase). The network is out of scope for v1.

---

## 5. Libratus — Brown & Sandholm 2017

Beat pros at HU NLHE. Three components:

1. **Blueprint strategy.** Pre-compute a coarse-abstraction CFR solution to the whole game.
2. **Subgame solving.** When we reach a node off the blueprint, solve the subgame exactly with safe re-solving (Burch et al. 2014 — guarantees we don't make the blueprint less safe).
3. **Self-improver.** Detect opponent's bet-size deviations during the match and add those sizings to the blueprint between sessions.

**Architectural lesson:** blueprint + on-the-fly refinement is the right pattern for large games.

---

## 6. Pluribus — Brown & Sandholm 2019

Beat pros at **6-max NLHE**. Multi-player is theoretically not solvable to Nash, so:

- **Trained via self-play with MCCFR**, no human data.
- **Depth-limited search** at decision time, but the leaves use *multiple "continuation strategies"* (4 of them: nominal, bias toward fold, bias toward call, bias toward raise) to handle the fact that opponents might play any of several strategies.
- **Action abstraction**: only a few bet sizes considered.
- **Asymmetric search**: only the bot does live search; opponents are assumed to play the blueprint.

**Lesson for our bot:** stay close to GTO, search at decision time when stakes are high, use a small set of bet sizes.

---

## 7. The Exploitative Layer — Where The "Human Element" Lives

None of the above papers do explicit opponent modeling — they win by being *unexploitable enough* that humans can't figure them out. But for our project we want **deliberate exploitation**, layered on top of GTO:

**Approach:** Bayesian posterior over opponent type, plus per-stat deviations.

- Track per-villain HUD stats with empirical Bayes shrinkage to a prior (so we don't over-react to small samples).
- Maintain posterior `P(type | observations)` over a small set of archetypes (TAG, LAG, fish, nit, maniac).
- For each archetype, precompute a **best-response strategy** vs. that archetype's typical play.
- Mix: `π_play = (1 − λ) · π_GTO + λ · Σ_t P(type=t | obs) · π_BR(t)`
- λ scales with confidence (sample size, stat stability) and shrinks vs. unknown opponents.

This is the safety mechanism: if we're wrong about the opponent, we fall back toward GTO and can't be counter-exploited.

**Files:** `pokerbot/opponent/`, `pokerbot/policy/hybrid.py`.

---

## Roadmap mapped to papers

| Phase | Implements | Tests against |
|-------|-----------|---------------|
| 1 | Vanilla CFR | Kuhn poker known Nash |
| 2 | MCCFR + CFR+ | Leduc poker (low exploitability) |
| 3 | Exploitability calculator | Both above |
| 4 | Opponent modeling | Synthetic archetype opponents |
| 5 | Hybrid GTO + exploit | Self-play tournament |
| 6 (future) | Depth-limited search | NLHE abstractions |

---

## Why Kuhn poker first

Kuhn poker (Kuhn 1950) is the smallest interesting poker game:
- 3-card deck (J, Q, K), 1 card each, 1 betting round, ante 1.
- 12 information sets total. Trivial to enumerate.
- **Nash equilibrium known analytically**: P1 bets J with prob `α ∈ [0, 1/3]`, never bets Q, bets K with prob `3α`. P1 calls Q-bet with prob `α + 1/3`. P2 has fixed Nash: bluffs J with `1/3`, calls K-bet always, etc.

If our CFR doesn't find this, *something* is wrong, and we know exactly what the answer should be.

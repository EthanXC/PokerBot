"""Visualize what GMM/EM discovered as opponent archetypes.

Prints the learned cluster centroids alongside the hand-coded archetypes
for comparison. Use this slide for the presentation.
"""
from __future__ import annotations

from pokerbot.cache import cached_learned_archetypes
from pokerbot.opponent import STAT_NAMES


def main() -> None:
    print("=" * 70)
    print("Unsupervised opponent archetype discovery (GMM + EM)")
    print("=" * 70)

    arch = cached_learned_archetypes(n_sessions=200, hands_per_session=80)

    print(f"\nFitted K = {arch.n_components} clusters from raw stat vectors.")
    print(f"Features: {STAT_NAMES}\n")

    print(f"{'cluster':>8}  {'weight':>7}   "
          + "  ".join(f"{n:>6}" for n in STAT_NAMES)
          + f"   {'exploit':>8}   {'closest archetype':<18}")
    print("-" * 80)

    expl = arch.exploitability_score_per_cluster()
    # Heuristic mapping from cluster characteristics to a poker-archetype name.
    def _label_cluster(means):
        vpip, pfr, agg, wtsd = means
        # Aggression-dominated
        if agg > 0.25 and pfr > 0.25:
            return "maniac / LAG"
        if pfr > 0.25:
            return "TAG"
        if wtsd > 0.55 and agg < 0.15:
            return "calling station"
        if agg < 0.10 and pfr < 0.15:
            return "nit / passive"
        return "balanced"

    for k in range(arch.n_components):
        means = arch.cluster_means[k]
        weight = arch.cluster_weights[k]
        score = expl[k]
        label = _label_cluster(means)
        print(f"{k:>8}  {weight:>7.3f}   "
              + "  ".join(f"{m:>6.3f}" for m in means)
              + f"   {score:>8.3f}   {label}")

    print()
    print("How the bot uses this:")
    print("  For each opponent, compute their (VPIP, PFR, AGG, WTSD) vector.")
    print("  GMM gives posterior P(z=k | stats) over clusters.")
    print("  Bot's deviation_strength is scaled by the weighted exploitability score.")
    print("  Result: more aggressive bluff-catching against 'maniac' clusters,")
    print("          tighter play against 'nit' clusters.")


if __name__ == "__main__":
    main()

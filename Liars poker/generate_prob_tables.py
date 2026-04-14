"""
generate_prob_tables.py

Generates a line-graph figure of poker hand probabilities for n=5..25.
Replaces the LaTeX table approach with a cleaner matplotlib figure.

Outputs (relative to this directory):
  figures/hand_probabilities.pdf   — main figure for inclusion in the paper
  figures/hand_probabilities.json  — raw data for reference
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poker_math_exact import get_hand_probabilities, HAND_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(HERE, "figures")

N_VALUES = list(range(5, 26))

# All hands except Royal Flush (too rare to be visible at useful scale)
HANDS = [
    "High Card",
    "Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
]

COLORS = plt.cm.tab10.colors


JSON_CACHE = os.path.join(FIGURES_DIR, "hand_probabilities.json")


def compute_all_probs(n_samples=3_000_000):
    """Compute probabilities for all n, using the JSON cache if available."""
    if os.path.exists(JSON_CACHE):
        print(f"  Loading cached probabilities from {JSON_CACHE}")
        with open(JSON_CACHE) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    print("  Cache not found — running Monte Carlo (this takes a few minutes)...")
    data = {}
    for n in N_VALUES:
        print(f"  n={n}...", end=" ", flush=True)
        data[n] = get_hand_probabilities(n, n_samples=n_samples)
        print("done")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(JSON_CACHE, "w") as f:
        json.dump({str(n): data[n] for n in N_VALUES}, f, indent=2)
    print(f"  Saved cache: {JSON_CACHE}")
    return data


def plot(data, output_path):
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, hand in enumerate(HANDS):
        probs = [data[n][hand] * 100 for n in N_VALUES]
        ax.plot(N_VALUES, probs, marker="o", markersize=3,
                label=hand, color=COLORS[i], linewidth=1.8)

    ax.set_xlabel("Pool size $n$ (cards)", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_title(
        "Blind hand probability: best 5-card hand from an $n$-card pool\n"
        r"(exact for $n\in\{5,6,7\}$; Monte Carlo $N=3\times10^6$ for $n\geq 8$)",
        fontsize=11,
    )
    ax.set_xticks(range(5, 21))
    ax.set_xlim(5, 20)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# Hand order from weakest to strongest (for stacking)
HANDS_ORDERED = [
    "High Card", "Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Royal Flush",
]


def plot_distribution(data, output_path):
    """Stacked area chart of P(best == X) for each hand type vs n."""
    fig, ax = plt.subplots(figsize=(11, 6))

    probs_matrix = np.array(
        [[data[n][hand] * 100 for n in N_VALUES] for hand in HANDS_ORDERED]
    )
    ax.stackplot(N_VALUES, probs_matrix, labels=HANDS_ORDERED,
                 colors=list(plt.cm.tab10.colors[:10]))

    ax.set_xlabel("Pool size $n$ (cards)", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_title(
        "Distribution of pool hand type as a function of pool size $n$\n"
        r"(exact for $n\in\{5,6,7\}$; Monte Carlo $N=3\times10^6$ for $n\geq 8$)",
        fontsize=11,
    )
    ax.set_xticks(range(5, 21))
    ax.set_xlim(5, 20)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8, reverse=True)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_at_least(data, output_path):
    """Line chart of P(best >= X) for each hand type (excluding High Card) vs n."""
    fig, ax = plt.subplots(figsize=(11, 6))

    # Skip High Card: P(best >= High Card) = 1 always, uninformative
    at_least_hands = HANDS_ORDERED[1:]

    for i, hand in enumerate(at_least_hands):
        hand_rank = HANDS_ORDERED.index(hand)
        at_least_probs = []
        for n in N_VALUES:
            p = sum(data[n][HAND_NAMES[j]] for j in range(hand_rank, 10))
            at_least_probs.append(p * 100)
        ax.scatter(N_VALUES, at_least_probs,
                   color=COLORS[i + 1], s=25, zorder=3,
                   label=f"$\\geq$ {hand}")

    ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2, alpha=0.55,
               label="50% threshold")

    ax.set_xlabel("Pool size $n$ (cards)", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_title(
        r"$P(\geq T,\, n)$: probability the pool best hand is at least strength $T$"
        "\n"
        r"(exact for $n\in\{5,6,7\}$; Monte Carlo $N=3\times10^6$ for $n\geq 8$)",
        fontsize=11,
    )
    ax.set_xticks(range(5, 21))
    ax.set_xlim(5, 20)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Computing/loading probabilities for n=5..25...")
    data = compute_all_probs()

    # Generate figures
    plot(data, os.path.join(FIGURES_DIR, "hand_probabilities.pdf"))
    plot_distribution(data, os.path.join(FIGURES_DIR, "hand_distribution.pdf"))
    plot_at_least(data, os.path.join(FIGURES_DIR, "hand_at_least.pdf"))
    print("Done.")


if __name__ == "__main__":
    main()

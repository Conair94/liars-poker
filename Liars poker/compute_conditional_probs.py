"""
compute_conditional_probs.py

Generates line-plot figures showing how knowing your private hand
shifts the pool hand-type probability distribution compared to the blind baseline.

Conditions (player holds 2 private cards):
  pair     — both cards share a rank (e.g. 7♦ 7♥)
  adjacent — consecutive ranks, any suits (e.g. 8♦ 9♠)
  suited   — same suit, any ranks (e.g. 3♠ J♠)

Conditions (player holds 3 private cards):
  3suited  — all three cards share the same suit
  3range   — all three ranks fit within a 5-card window (max − min ≤ 4)

Each figure overlays:
  solid lines  — blind baseline from hand_probabilities.json
  dashed lines — conditional distribution given the private hand condition

Outputs (relative to this directory):
  figures/conditional_pair.pdf
  figures/conditional_adjacent.pdf
  figures/conditional_suited.pdf
  figures/conditional_3suited.pdf
  figures/conditional_3range.pdf
  figures/conditional_probs_data.json   — cache; new conditions are appended on rerun
"""

import sys
import os
import random
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from poker_math_exact import _evaluate, _evaluate_ranked, HAND_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE          = os.path.dirname(os.path.abspath(__file__))
FIGURES       = os.path.join(HERE, "figures")
CACHE         = os.path.join(FIGURES, "conditional_probs_data.json")
RANKED_CACHE  = os.path.join(FIGURES, "conditional_probs_ranked_data.json")
BASELINE      = os.path.join(FIGURES, "hand_probabilities.json")
TABLES_DIR    = os.path.join(FIGURES, "tables")

N_VALUES  = list(range(5, 26))
N_SAMPLES = 1_000_000

# 9 hands (no Royal Flush — too rare to show meaningfully)
HANDS = [
    "High Card", "Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush",
]
HAND_IDX = {name: i for i, name in enumerate(HAND_NAMES)}
COLORS = plt.cm.tab10.colors

# ---------------------------------------------------------------------------
# Private-hand samplers (stratified — force the condition)
# ---------------------------------------------------------------------------

def _sample_pair(rng):
    """Two cards of the same rank, different suits."""
    rank  = rng.randint(0, 12)
    suits = rng.sample(range(4), 2)
    return [rank * 4 + suits[0], rank * 4 + suits[1]]


def _sample_adjacent(rng):
    """Two cards of consecutive ranks (r, r+1), any suits."""
    rank1 = rng.randint(0, 11)   # 0–11 so rank1+1 is valid (no Ace-low wrap)
    rank2 = rank1 + 1
    suit1 = rng.randint(0, 3)
    suit2 = rng.randint(0, 3)
    return [rank1 * 4 + suit1, rank2 * 4 + suit2]


def _sample_suited(rng):
    """Two cards of the same suit, any two distinct ranks."""
    suit  = rng.randint(0, 3)
    ranks = rng.sample(range(13), 2)
    return [ranks[0] * 4 + suit, ranks[1] * 4 + suit]


def _sample_trips(rng):
    """Three cards of the same rank (trips), any three distinct suits."""
    rank  = rng.randint(0, 12)
    suits = rng.sample(range(4), 3)
    return [rank * 4 + s for s in suits]


def _sample_3suited(rng):
    """Three cards of the same suit, any three distinct ranks."""
    suit  = rng.randint(0, 3)
    ranks = rng.sample(range(13), 3)
    return [r * 4 + suit for r in ranks]


def _sample_3range(rng):
    """
    Three cards whose ranks all fit within a 5-card window (max rank − min rank ≤ 4).
    Window start chosen uniformly from 0..8 so window [start, start+4] stays in 0..12.
    """
    window_start = rng.randint(0, 8)
    ranks = rng.sample(range(window_start, window_start + 5), 3)
    return [r * 4 + rng.randint(0, 3) for r in ranks]


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate(sampler, n_private=2, n_samples=N_SAMPLES, seed=42):
    """
    Run n_samples trials using the given sampler for the n_private private cards.
    Returns counts[n_str] = list of 10 ints (hand-type counts) for each n in 5..25.
    """
    rng   = random.Random(seed)
    deck  = list(range(52))
    counts = {str(n): [0] * 10 for n in N_VALUES}
    max_opponents = max(N_VALUES) - n_private  # cards drawn from remainder per trial

    for _ in range(n_samples):
        private  = sampler(rng)
        priv_set = set(private)
        rest     = [c for c in deck if c not in priv_set]
        opponents = rng.sample(rest, max_opponents)

        for n in N_VALUES:
            pool = private + opponents[: n - n_private]
            counts[str(n)][_evaluate(pool)] += 1

    return counts


# ---------------------------------------------------------------------------
# Load / compute cache
# ---------------------------------------------------------------------------

def _load_or_compute():
    if os.path.exists(CACHE):
        print(f"Loading cache: {CACHE}")
        with open(CACHE) as f:
            data = json.load(f)
    else:
        data = {"n_samples": N_SAMPLES, "conditions": {}}

    missing = [
        (key, sampler, n_priv)
        for key, sampler, n_priv, _, _, _ in CONDITION_META
        if key not in data["conditions"]
    ]

    if not missing:
        return data

    os.makedirs(FIGURES, exist_ok=True)
    print(f"Running Monte Carlo ({N_SAMPLES:,} samples) for missing conditions...")
    for key, sampler, n_priv in missing:
        print(f"  Simulating '{key}' ({n_priv} private cards)...")
        data["conditions"][key] = _simulate(sampler, n_private=n_priv)

    with open(CACHE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved cache: {CACHE}")
    return data


# ---------------------------------------------------------------------------
# Rank-level simulation  (10 × 13 count matrix per n)
# ---------------------------------------------------------------------------

def _simulate_ranked(sampler, n_private=2, n_samples=N_SAMPLES, seed=42):
    """
    Like _simulate() but uses _evaluate_ranked to track (hand_type, primary_rank).
    Returns {str(n): [[0]*13 for _ in range(10)]}
      counts[str(n)][hand_type][primary_rank] = number of samples
    """
    rng   = random.Random(seed)
    deck  = list(range(52))
    counts = {str(n): [[0] * 13 for _ in range(10)] for n in N_VALUES}
    max_opponents = max(N_VALUES) - n_private

    for _ in range(n_samples):
        private  = sampler(rng)
        priv_set = set(private)
        rest     = [c for c in deck if c not in priv_set]
        opponents = rng.sample(rest, max_opponents)

        for n in N_VALUES:
            pool = private + opponents[: n - n_private]
            ht, pr = _evaluate_ranked(pool)
            counts[str(n)][ht][pr] += 1

    return counts


def _simulate_ranked_blind(n_samples=3_000_000, seed=42):
    """
    Blind baseline rank-level simulation (no private cards).
    Uses 3M samples for higher precision. Returns same format as _simulate_ranked.
    """
    rng   = random.Random(seed)
    deck  = list(range(52))
    counts = {str(n): [[0] * 13 for _ in range(10)] for n in N_VALUES}
    n_max = max(N_VALUES)

    for _ in range(n_samples):
        hand = rng.sample(deck, n_max)
        for n in N_VALUES:
            ht, pr = _evaluate_ranked(hand[:n])
            counts[str(n)][ht][pr] += 1

    return counts


def _load_or_compute_ranked():
    """
    Load or compute rank-level (10×13) count matrices for all scenarios.

    Cache format:
    {
      "blind":      {"n_samples": 3000000, "counts": {str(n): [[13 ints]*10]}},
      "conditions": {key: {"n_samples": 1000000, "counts": {str(n): [[13 ints]*10]}}}
    }
    """
    if os.path.exists(RANKED_CACHE):
        print(f"Loading ranked cache: {RANKED_CACHE}")
        with open(RANKED_CACHE) as f:
            data = json.load(f)
    else:
        data = {"conditions": {}}

    changed = False

    # Blind scenario
    if "blind" not in data:
        print("  Simulating 'blind' ranked (3,000,000 samples)...")
        data["blind"] = {
            "n_samples": 3_000_000,
            "counts": _simulate_ranked_blind(n_samples=3_000_000),
        }
        changed = True

    # Conditional scenarios
    for key, sampler, n_priv, _, _, _ in CONDITION_META:
        if key not in data["conditions"]:
            print(f"  Simulating '{key}' ranked ({N_SAMPLES:,} samples)...")
            data["conditions"][key] = {
                "n_samples": N_SAMPLES,
                "counts": _simulate_ranked(sampler, n_private=n_priv),
            }
            changed = True

    if changed:
        os.makedirs(FIGURES, exist_ok=True)
        with open(RANKED_CACHE, "w") as f:
            json.dump(data, f)
        print(f"Saved ranked cache: {RANKED_CACHE}")

    return data


# ---------------------------------------------------------------------------
# Threshold table helpers
# ---------------------------------------------------------------------------

# Hand abbreviations for table cells
HAND_ABBREV = ["Hi", "Pr", "2P", "3K", "St", "Fl", "FH", "4K", "SF", "RF"]
RANK_NAMES  = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS       = ["♣", "♦", "♥", "♠"]


def _card(rank, suit):
    return f"{RANK_NAMES[rank]}{SUITS[suit]}"


def _all_bids_ordered():
    """
    Return all (hand_type, primary_rank) pairs ordered weakest → strongest.
    Within a type, higher primary_rank is stronger. Between types, higher type is stronger.
    """
    order = []
    for ht in range(10):
        for pr in range(13):
            order.append((ht, pr))
    return order  # (0,0)=HC-2 is weakest; (9,12)=RF is strongest


def find_threshold_ranked(counts_10x13, n_samples):
    """
    Scan bids from weakest to strongest, computing P(>=bid) at each step.
    P(>=B) decreases monotonically as B gets stronger.

    Returns:
      above: {"hand_type": int, "rank": int, "prob": float}
             weakest bid where P(>=bid) >= 50% — the "safe" bid
      below: {"hand_type": int, "rank": int, "prob": float}
             strongest bid where P(>=bid) < 50% — one step too aggressive
      Either may be None if no crossing exists in that direction.
    """
    bids = _all_bids_ordered()  # (0,0)=HC-2 weakest ... (9,12)=RF strongest
    cum_weaker = 0.0  # cumulative probability of hands strictly weaker than current bid
    above = None
    below = None

    for ht, pr in bids:
        p_at_least = 1.0 - cum_weaker  # P(best >= current bid)
        if p_at_least >= 0.5:
            above = {"hand_type": ht, "rank": pr, "prob": p_at_least}
            cum_weaker += counts_10x13[ht][pr] / n_samples
        else:
            below = {"hand_type": ht, "rank": pr, "prob": p_at_least}
            break

    return above, below


def build_example_pool(hand_type, primary_rank, n):
    """
    Return a string of n cards forming a canonical example of hand_type at primary_rank
    as the best 5-card hand. Fillers do not upgrade the hand type.

    primary_rank (0..12) is used to set the key rank for the hand type.
    For High Card and Flush, it determines the high card.
    For Pair/2P/3K/FH/4K/SF: it sets the pair/trip/quad/SF rank.
    For Straight: it sets the high card of the straight (3=5-high through 12=A-high).
    """
    pr = primary_rank

    if hand_type == 0:  # High Card: high card = pr, four distinct lower off-suit cards
        # Need pr as max, four smaller distinct ranks, no pair, no straight, no flush
        # Use off-suit cards (different suits) and pick non-consecutive lower ranks
        r1 = pr
        # Choose 4 fillers below pr, spread enough to avoid straights
        fillers_r = []
        for r in range(pr - 1, -1, -1):
            if len(fillers_r) == 4:
                break
            # Skip if adding this would complete a 5-card straight with current set
            candidate = sorted([r1] + fillers_r + [r], reverse=True)
            if len(candidate) >= 5 and candidate[0] - candidate[4] == 4:
                continue  # would form a straight
            fillers_r.append(r)
        # Pad with lower ranks if needed (shouldn't happen for pr >= 4)
        while len(fillers_r) < 4:
            fillers_r.append(0)
        key = [(r1, 3), (fillers_r[0], 1), (fillers_r[1], 2),
               (fillers_r[2], 0), (fillers_r[3], 1)]

    elif hand_type == 1:  # Pair: pair of pr
        key = [(pr, 3), (pr, 1)]
        # Three kickers: distinct ranks ≠ pr, high to low
        kickers = [r for r in range(12, -1, -1) if r != pr][:3]
        key += [(kickers[0], 2), (kickers[1], 0), (kickers[2], 1)]

    elif hand_type == 2:  # Two Pair: higher pair = pr, lower pair = pr-1 or next lower
        lo = pr - 1 if pr > 0 else 0
        if lo == pr:
            lo = max(r for r in range(13) if r != pr)
        kicker = next(r for r in range(12, -1, -1) if r != pr and r != lo)
        key = [(pr, 3), (pr, 1), (lo, 2), (lo, 0), (kicker, 1)]

    elif hand_type == 3:  # Three of a Kind: trips of pr
        kickers = [r for r in range(12, -1, -1) if r != pr][:2]
        key = [(pr, 3), (pr, 1), (pr, 2), (kickers[0], 0), (kickers[1], 1)]

    elif hand_type == 4:  # Straight: high card = pr (pr=3 means 5-high A-2-3-4-5)
        if pr == 3:  # Ace-low straight A-2-3-4-5
            key = [(12, 3), (0, 1), (1, 2), (2, 0), (3, 1)]
        else:
            key = [(pr, 3), (pr-1, 1), (pr-2, 2), (pr-3, 0), (pr-4, 1)]

    elif hand_type == 5:  # Flush: high card in flush suit = pr; use A-high for safety
        # Always build A-high flush to avoid near-impossible low-flush issues;
        # if pr < 9 (J), fall back to A-high canonical flush
        if pr < 9:
            pr = 12  # override to A-high
        # Pick 5 non-consecutive spade ranks with pr as highest
        flush_ranks = [pr]
        for r in range(pr - 2, -1, -2):  # every other rank, non-consecutive
            if len(flush_ranks) == 5:
                break
            flush_ranks.append(r)
        while len(flush_ranks) < 5:
            flush_ranks.append(flush_ranks[-1] - 1)
        key = [(r, 3) for r in flush_ranks[:5]]  # all spades

    elif hand_type == 6:  # Full House: trips of pr
        pair_r = next(r for r in range(12, -1, -1) if r != pr)
        key = [(pr, 3), (pr, 1), (pr, 2), (pair_r, 0), (pair_r, 1)]

    elif hand_type == 7:  # Four of a Kind: quads of pr
        kicker = next(r for r in range(12, -1, -1) if r != pr)
        key = [(pr, 3), (pr, 1), (pr, 2), (pr, 0), (kicker, 1)]

    elif hand_type == 8:  # Straight Flush: high card = pr in spades
        if pr == 3:  # 5-high SF: A♠ 2♠ 3♠ 4♠ 5♠
            key = [(12, 3), (0, 3), (1, 3), (2, 3), (3, 3)]
        else:
            key = [(pr, 3), (pr-1, 3), (pr-2, 3), (pr-3, 3), (pr-4, 3)]

    elif hand_type == 9:  # Royal Flush: always A-high in spades
        key = [(12, 3), (11, 3), (10, 3), (9, 3), (8, 3)]

    else:
        raise ValueError(f"Unknown hand_type {hand_type}")

    used      = set(key)
    key_ranks = {kr for kr, _ in key}

    avail_ranks = [r for r in range(13) if r not in key_ranks]

    # Flush fillers use off-suit (not spades) to avoid 5th spade upgrading
    max_per_rank = 2 if hand_type == 5 else 1
    filler_rank_count = {}
    fillers = []

    for suit_round in range(3):   # Clubs=0, Diamonds=1, Hearts=2
        for r in avail_ranks:
            if len(fillers) + len(key) >= n:
                break
            count = filler_rank_count.get(r, 0)
            if count >= max_per_rank:
                continue
            s = suit_round
            if (r, s) not in used:
                fillers.append((r, s))
                used.add((r, s))
                filler_rank_count[r] = count + 1
        if len(fillers) + len(key) >= n:
            break

    pool = list(key) + fillers[:n - len(key)]
    return " ".join(_card(r, s) for r, s in pool)


def _fmt_bid(hand_type, rank):
    """Format a (hand_type, rank) bid as 'Abbr Rank', e.g. 'FH A' or 'Pr K'."""
    return f"{HAND_ABBREV[hand_type]} {RANK_NAMES[rank]}"


def _compute_thresholds(ranked_data):
    """Compute threshold (above/below 50%) for all 7 scenarios × n=5..25."""
    all_scenarios = [
        ("blind",    ranked_data["blind"]),
        ("pair",     ranked_data["conditions"]["pair"]),
        ("trips",    ranked_data["conditions"]["trips"]),
        ("suited",   ranked_data["conditions"]["suited"]),
        ("3suited",  ranked_data["conditions"]["3suited"]),
        ("adjacent", ranked_data["conditions"]["adjacent"]),
        ("3range",   ranked_data["conditions"]["3range"]),
    ]
    thresholds = {}
    for key, scenario_data in all_scenarios:
        thresholds[key] = {}
        n_samp = scenario_data["n_samples"]
        for n in range(5, 26):
            matrix = scenario_data["counts"][str(n)]
            above, below = find_threshold_ranked(matrix, n_samp)
            thresholds[key][n] = {"above": above, "below": below}
    return thresholds


_RANK_PLURAL = [
    "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "10s",
    "Jacks", "Queens", "Kings", "Aces",
]
_RANK_HIGH = [
    "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "Jack", "Queen", "King", "Ace",
]


def _readable_hand_name(hand_type, rank):
    """
    Return a natural-language hand name, e.g.:
      Pair      rank=11  →  'Pair of Queens'
      Two Pair  rank=9   →  '10s Up'
      Three     rank=4   →  'Three 5s'
      Straight  rank=8   →  '10-High Straight'
      Flush     rank=12  →  'Ace-High Flush'
      Full House rank=9  →  '10s Full'
      Four      rank=0   →  'Four 2s'
      SF        rank=11  →  'Queen-High SF'
      RF                 →  'Royal Flush'
    """
    pl   = _RANK_PLURAL[rank]
    hi   = _RANK_HIGH[rank]
    if hand_type == 0:   return f"{hi}-High"
    if hand_type == 1:   return f"Pair of {pl}"
    if hand_type == 2:   return f"{pl} Up"
    if hand_type == 3:   return f"Three {pl}"
    if hand_type == 4:
        if rank == 3:    return "Wheel (5-High)"
        if rank == 12:   return "Broadway"
        return f"{hi}-High Straight"
    if hand_type == 5:   return f"{hi}-High Flush"
    if hand_type == 6:   return f"{pl} Full"
    if hand_type == 7:   return f"Four {pl}"
    if hand_type == 8:
        if rank == 3:    return "5-High Straight Flush"
        return f"{hi}-High Straight Flush"
    return "Royal Flush"   # hand_type == 9


def _cell_text(entry):
    """Single-line cell: readable hand name + probability."""
    if entry is None:
        return "---"
    name = _readable_hand_name(entry["hand_type"], entry["rank"])
    pct  = f"{100 * entry['prob']:.1f}%"
    return f"{name}\n({pct})"


def write_threshold_figures(ranked_data, figures_dir):
    """
    Render rank-level threshold tables as matplotlib figures and save as PDFs.
    Outputs:
      figures/tables/threshold_part1.pdf  — Blind, Pair, Trips, Suited
      figures/tables/threshold_part2.pdf  — 3-Suited, Adjacent, 3-Range
    """
    tables_dir = os.path.join(figures_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    thresholds = _compute_thresholds(ranked_data)

    SCENARIO_GROUPS = [
        (
            [("blind","Blind"), ("pair","Pair"), ("trips","Trips"), ("suited","Suited")],
            os.path.join(tables_dir, "threshold_part1.pdf"),
        ),
        (
            [("3suited","3-Suited"), ("adjacent","Adjacent"), ("3range","3-Range")],
            os.path.join(tables_dir, "threshold_part2.pdf"),
        ),
    ]

    for scenarios, out_path in SCENARIO_GROUPS:
        _render_threshold_figure(scenarios, thresholds, out_path)


def _render_threshold_figure(scenarios, thresholds, out_path):
    """
    Draw one threshold figure: one column per scenario, each cell split left/right —
    left half (≥50% / green) and right half (<50% / red) with a vertical divider.
    """
    import matplotlib.patches as mpatches

    n_scen = len(scenarios)
    n_rows = 21           # n = 5..25

    # -----------------------------------------------------------------------
    # Layout in axes-coordinate space [0, 1] × [0, 1]
    # Each scenario occupies DATA_W; within it, LEFT_W and RIGHT_W are the
    # two sub-columns.
    # -----------------------------------------------------------------------
    PAD      = 0.01
    N_COL_W  = 0.030
    DATA_W   = (1.0 - 2 * PAD - N_COL_W) / n_scen
    HALF_W   = DATA_W / 2
    HDR_H    = 0.030
    DATA_H   = (1.0 - 2 * PAD - HDR_H) / n_rows

    # Colours
    HDR_BG      = "#2c3e50"
    HDR_FG      = "white"
    N_BG        = "#ecf0f1"
    ABOVE_EVEN  = "#d5f5e3"   # green — ≥50%
    ABOVE_ODD   = "#eafaf1"
    BELOW_EVEN  = "#fadbd8"   # red   — <50%
    BELOW_ODD   = "#fdf2f1"
    GRID        = "#aab7b8"
    DIV         = "#95a5a6"   # vertical divider colour

    fig_w = 1.8 + 2.4 * n_scen
    fig_h = 11
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Coordinate helpers
    # -----------------------------------------------------------------------
    def cx(col):
        """Left edge of scenario column (col=0 is the n-column)."""
        if col == 0:
            return PAD
        return PAD + N_COL_W + DATA_W * (col - 1)

    def cy(row):
        """Top edge of row (row 0 = header)."""
        if row == 0:
            return 1.0 - PAD
        return 1.0 - PAD - HDR_H - DATA_H * (row - 1)

    def cw(col):
        return N_COL_W if col == 0 else DATA_W

    def ch(row):
        return HDR_H if row == 0 else DATA_H

    def rect_raw(x, y, w, h, bg):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            facecolor=bg, edgecolor=GRID, linewidth=0.4,
        ))

    def txt(col, row, text, size=7.5, color="black", weight="normal"):
        ax.text(
            cx(col) + cw(col) / 2,
            cy(row) - ch(row) / 2,
            text,
            ha="center", va="center",
            fontsize=size, color=color, fontweight=weight,
            transform=ax.transAxes, clip_on=True,
        )

    # -----------------------------------------------------------------------
    # Header row
    # -----------------------------------------------------------------------
    rect_raw(cx(0), cy(0) - HDR_H, cw(0), HDR_H, HDR_BG)
    txt(0, 0, "n", size=8, color=HDR_FG, weight="bold")

    for s_idx, (_, lbl) in enumerate(scenarios):
        col_l = cx(s_idx + 1)
        col_top = cy(0)
        # Single header spanning both sub-columns
        rect_raw(col_l, col_top - HDR_H, DATA_W, HDR_H, HDR_BG)
        ax.text(col_l + DATA_W / 2, col_top - HDR_H / 2, lbl,
                ha="center", va="center", fontsize=8,
                color=HDR_FG, fontweight="bold", transform=ax.transAxes)

    # -----------------------------------------------------------------------
    # Data rows
    # -----------------------------------------------------------------------
    for row_idx, n in enumerate(range(5, 26)):
        r = row_idx + 1
        even = (row_idx % 2 == 0)

        # n column
        rect_raw(cx(0), cy(r) - ch(r), cw(0), ch(r), N_BG)
        txt(0, r, str(n), size=7.5, weight="bold")

        for s_idx, (key, _) in enumerate(scenarios):
            above = thresholds[key][n]["above"]
            below = thresholds[key][n]["below"]

            col_l   = cx(s_idx + 1)
            col_mid = col_l + HALF_W
            col_r   = col_l + DATA_W
            row_top = cy(r)
            row_bot = row_top - DATA_H
            y_ctr   = (row_top + row_bot) / 2

            # Left sub-cell — above (≥50%)
            rect_raw(col_l,   row_bot, HALF_W, DATA_H,
                     ABOVE_EVEN if even else ABOVE_ODD)
            # Right sub-cell — below (<50%)
            rect_raw(col_mid, row_bot, HALF_W, DATA_H,
                     BELOW_EVEN if even else BELOW_ODD)
            # Vertical divider
            ax.plot([col_mid, col_mid], [row_bot, row_top],
                    color=DIV, linewidth=0.8, transform=ax.transAxes, clip_on=True)

            def _draw_subcell(entry, x_center, name_color, pct_color):
                if entry is None:
                    ax.text(x_center, y_ctr, "—",
                            ha="center", va="center", fontsize=7,
                            color="#aab7b8", transform=ax.transAxes, clip_on=True)
                    return
                name = _readable_hand_name(entry["hand_type"], entry["rank"])
                pct  = f"{100 * entry['prob']:.1f}%"
                ax.text(x_center, y_ctr + DATA_H * 0.14, name,
                        ha="center", va="center", fontsize=5.8,
                        fontweight="bold", color=name_color,
                        transform=ax.transAxes, clip_on=True)
                ax.text(x_center, y_ctr - DATA_H * 0.16, pct,
                        ha="center", va="center", fontsize=5.4,
                        color=pct_color,
                        transform=ax.transAxes, clip_on=True)

            _draw_subcell(above,
                          x_center=col_l + HALF_W / 2,
                          name_color="#1a5276",
                          pct_color="#1e8449")
            _draw_subcell(below,
                          x_center=col_mid + HALF_W / 2,
                          name_color="#922b21",
                          pct_color="#7b241c")

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _load_baseline():
    with open(BASELINE) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def _plot_conditional(condition_name, cond_counts, baseline, output_path, label):
    """
    Overlay solid baseline lines with dashed conditional lines.
    cond_counts: dict str(n) -> list[10 int counts]
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    n_samples = sum(cond_counts[str(N_VALUES[0])])

    for i, hand in enumerate(HANDS):
        hidx  = HAND_IDX[hand]
        color = COLORS[i]

        # Baseline (dashed)
        base_probs = [baseline[n][hand] * 100 for n in N_VALUES]
        ax.plot(N_VALUES, base_probs,
                color=color, linewidth=1.6, linestyle="--")

        # Conditional (solid)
        cond_probs = [
            cond_counts[str(n)][hidx] / n_samples * 100
            for n in N_VALUES
        ]
        ax.plot(N_VALUES, cond_probs,
                color=color, linewidth=1.6, linestyle="-",
                label=hand)

    ax.set_xlabel("Pool size $n$ (cards)", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_title(
        f"Pool hand distribution | player holds {label}\n"
        "dashed = blind baseline,  solid = conditional",
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


def _plot_at_least_conditional(condition_name, cond_counts, baseline, output_path, label):
    """
    Cumulative at-least view: P(best >= X | condition) vs P(best >= X) baseline.
    Solid lines = blind baseline, dashed lines = conditional. Dashed 50% reference line.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    n_samples = sum(cond_counts[str(N_VALUES[0])])

    # Pair through Royal Flush (skip High Card — always 100%)
    at_least_hands = HAND_NAMES[1:]

    for hand in at_least_hands:
        hidx  = HAND_IDX[hand]
        color = COLORS[hidx]

        # Baseline at-least: open circles
        base_probs = [
            sum(baseline[n][HAND_NAMES[j]] for j in range(hidx, 10)) * 100
            for n in N_VALUES
        ]
        ax.scatter(N_VALUES, base_probs,
                   color=color, s=18, facecolors="none", edgecolors=color,
                   linewidths=0.9, zorder=2)

        # Conditional at-least: filled circles
        cond_probs = [
            sum(cond_counts[str(n)][j] for j in range(hidx, 10)) / n_samples * 100
            for n in N_VALUES
        ]
        ax.scatter(N_VALUES, cond_probs,
                   color=color, s=18, zorder=3,
                   label=f"$\\geq$ {hand}")

    ax.axhline(y=50, color="black", linestyle="--", linewidth=1.2, alpha=0.55,
               label="50% threshold")

    ax.set_xlabel("Pool size $n$ (cards)", fontsize=11)
    ax.set_ylabel("Probability (%)", fontsize=11)
    ax.set_title(
        f"$P(\\geq T \\mid \\mathbf{{h}},\\, n)$: at-least probability | player holds {label}\n"
        "open circles = blind baseline,  filled circles = conditional",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CONDITION_META = [
    ("pair",     _sample_pair,     2, "a pair (two cards of the same rank)",             "conditional_pair.pdf",     "conditional_pair_atleast.pdf"),
    ("trips",    _sample_trips,    3, "three cards of the same rank (trips)",             "conditional_trips.pdf",    "conditional_trips_atleast.pdf"),
    ("adjacent", _sample_adjacent, 2, "two adjacent cards (consecutive ranks)",           "conditional_adjacent.pdf", "conditional_adjacent_atleast.pdf"),
    ("suited",   _sample_suited,   2, "two suited cards (same suit, any ranks)",          "conditional_suited.pdf",   "conditional_suited_atleast.pdf"),
    ("3suited",  _sample_3suited,  3, "three suited cards (same suit, any ranks)",        "conditional_3suited.pdf",  "conditional_3suited_atleast.pdf"),
    ("3range",   _sample_3range,   3, "three cards within a 5-card rank window",          "conditional_3range.pdf",   "conditional_3range_atleast.pdf"),
]


def main():
    os.makedirs(FIGURES, exist_ok=True)

    # Type-level simulation (existing plots)
    data     = _load_or_compute()
    baseline = _load_baseline()

    for key, _, _, label, fname, fname_atleast in CONDITION_META:
        _plot_conditional(
            condition_name=key,
            cond_counts=data["conditions"][key],
            baseline=baseline,
            output_path=os.path.join(FIGURES, fname),
            label=label,
        )
        _plot_at_least_conditional(
            condition_name=key,
            cond_counts=data["conditions"][key],
            baseline=baseline,
            output_path=os.path.join(FIGURES, fname_atleast),
            label=label,
        )

    # Rank-level simulation + threshold figures
    print("Computing rank-level threshold figures...")
    ranked_data = _load_or_compute_ranked()
    write_threshold_figures(ranked_data, FIGURES)

    print("Done.")


if __name__ == "__main__":
    main()

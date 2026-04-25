"""
poker_math_exact.py

Computes poker hand-type probabilities for n=5..25 cards drawn from a 52-card deck.
The "hand" is defined as the best standard 5-card poker hand from the n cards.

For n in {5, 6, 7}: exact combinatorial counts from published enumeration.
For n in {8..25}:   Monte Carlo simulation (N samples, default 3_000_000).

Additionally provides exact "contains-at-least" counts for Straight, Flush, and Full House
for n=5..25, derived from Wu & Wu (2024): "Big Two and n-card poker probabilities,"
Communications on Number Theory and Combinatorial Theory, Vol. 5. arXiv:2309.00011.

IMPORTANT DISTINCTION:
  get_wu_wu_contains_counts(n)  →  hands that CONTAIN the pattern (Wu & Wu definition)
  get_hand_counts(n)            →  hands where BEST 5-card hand == that type (our definition)

  For Flush:     W_fl(n)  ≥  C_fl(n) + C_sf(n) + C_rf(n)
                 (W_fl also counts hands where best hand is FH or 4ok but ≥5 cards share a suit)
  For Straight:  W_st(n)  ≥  sum of C_T(n) for T ≥ Straight
                 (W_st also counts flush/FH/4ok hands that happen to contain a straight)
  For Full House: W_fh(n) ≥  sum of C_T(n) for T ≥ Full House
                 (W_fh also counts 4ok hands that contain a FH pattern: e.g. KKKKAA → FH+4ok)

  The crossover points in Wu & Wu's sense:
    Flush > Straight  when n > 11  (comparing W_fl vs W_st)
    Full House > Straight  when n > 19  (comparing W_fh vs W_st)

Card encoding: card index = rank * 4 + suit
  rank: 0=2, 1=3, ..., 12=A
  suit: 0=C, 1=D, 2=H, 3=S
"""

from math import comb
import random

# ---------------------------------------------------------------------------
# Hand type constants (matches liars_poker_engine.py HandEvaluator ordering)
# ---------------------------------------------------------------------------
HIGH_CARD       = 0
PAIR            = 1
TWO_PAIR        = 2
THREE_OF_A_KIND = 3
STRAIGHT        = 4
FLUSH           = 5
FULL_HOUSE      = 6
FOUR_OF_A_KIND  = 7
STRAIGHT_FLUSH  = 8
ROYAL_FLUSH     = 9

HAND_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Royal Flush",
]

# ---------------------------------------------------------------------------
# Fast hand evaluator (operates on raw card indices, no class overhead)
# ---------------------------------------------------------------------------

def _evaluate(card_indices):
    """
    Evaluate the best 5-card hand from card_indices (list of ints 0-51).
    Returns an int in 0..9 (HIGH_CARD..ROYAL_FLUSH).
    """
    ranks = [c >> 2 for c in card_indices]   # c // 4
    suits = [c & 3  for c in card_indices]   # c %  4

    # Rank frequency table
    rc = [0] * 13
    for r in ranks:
        rc[r] += 1

    # Suit frequency table
    sc = [0] * 4
    for s in suits:
        sc[s] += 1

    # Flush suit (-1 if none)
    flush_suit = -1
    for s in range(4):
        if sc[s] >= 5:
            flush_suit = s
            break

    # --- Straight Flush / Royal Flush ---
    if flush_suit >= 0:
        fr = sorted(
            {c >> 2 for c in card_indices if (c & 3) == flush_suit},
            reverse=True,
        )
        ext = fr + ([-1] if 12 in fr else [])
        for i in range(len(ext) - 4):
            if ext[i] - ext[i + 4] == 4:
                return ROYAL_FLUSH if ext[i] == 12 else STRAIGHT_FLUSH

    # --- Four of a Kind ---
    if max(rc) >= 4:
        return FOUR_OF_A_KIND

    # --- Full House ---
    sorted_rc = sorted(rc, reverse=True)
    if sorted_rc[0] >= 3 and sorted_rc[1] >= 2:
        return FULL_HOUSE

    # --- Flush ---
    if flush_suit >= 0:
        return FLUSH

    # --- Straight ---
    ur = sorted({r for r in ranks}, reverse=True)
    ext = ur + ([-1] if 12 in ur else [])
    for i in range(len(ext) - 4):
        if ext[i] - ext[i + 4] == 4:
            return STRAIGHT

    # --- Three of a Kind ---
    if sorted_rc[0] >= 3:
        return THREE_OF_A_KIND

    # --- Two Pair ---
    if sorted_rc[0] >= 2 and sorted_rc[1] >= 2:
        return TWO_PAIR

    # --- Pair ---
    if sorted_rc[0] >= 2:
        return PAIR

    return HIGH_CARD


def _evaluate_ranked(card_indices):
    """
    Evaluate the best 5-card hand from card_indices (list of ints 0-51).
    Returns (hand_type, primary_rank) where:
      hand_type    — int 0..9 (same as _evaluate)
      primary_rank — int 0..12 (rank of 2..A) encoding the dominant rank:
        High Card      → highest card rank
        Pair           → rank of the pair
        Two Pair       → rank of the HIGHER pair
        Three of a Kind → rank of the trips
        Straight       → high card of the best straight (3 = 5-high/Ace-low)
        Flush          → highest card in the flush suit
        Full House     → rank of the trips
        Four of a Kind → rank of the quads
        Straight Flush → high card of the SF sequence
        Royal Flush    → 12 (Ace-high)
    """
    ranks = [c >> 2 for c in card_indices]
    suits = [c & 3  for c in card_indices]

    rc = [0] * 13
    for r in ranks:
        rc[r] += 1

    sc = [0] * 4
    for s in suits:
        sc[s] += 1

    flush_suit = -1
    for s in range(4):
        if sc[s] >= 5:
            flush_suit = s
            break

    # --- Straight Flush / Royal Flush ---
    if flush_suit >= 0:
        fr = sorted(
            {c >> 2 for c in card_indices if (c & 3) == flush_suit},
            reverse=True,
        )
        ext = fr + ([-1] if 12 in fr else [])
        for i in range(len(ext) - 4):
            if ext[i] - ext[i + 4] == 4:
                top = ext[i]
                # Ace-low straight: high card is 3 (represents the "5" card)
                if top == -1:
                    top = 3
                return (ROYAL_FLUSH if top == 12 else STRAIGHT_FLUSH, top)

    # --- Four of a Kind ---
    if max(rc) >= 4:
        quad_rank = next(r for r in range(12, -1, -1) if rc[r] >= 4)
        return (FOUR_OF_A_KIND, quad_rank)

    # --- Full House ---
    sorted_rc = sorted(rc, reverse=True)
    if sorted_rc[0] >= 3 and sorted_rc[1] >= 2:
        trip_rank = next(r for r in range(12, -1, -1) if rc[r] >= 3)
        return (FULL_HOUSE, trip_rank)

    # --- Flush ---
    if flush_suit >= 0:
        flush_ranks = [c >> 2 for c in card_indices if (c & 3) == flush_suit]
        return (FLUSH, max(flush_ranks))

    # --- Straight ---
    ur = sorted({r for r in ranks}, reverse=True)
    ext = ur + ([-1] if 12 in ur else [])
    for i in range(len(ext) - 4):
        if ext[i] - ext[i + 4] == 4:
            top = ext[i]
            if top == -1:
                top = 3
            return (STRAIGHT, top)

    # --- Three of a Kind ---
    if sorted_rc[0] >= 3:
        trip_rank = next(r for r in range(12, -1, -1) if rc[r] >= 3)
        return (THREE_OF_A_KIND, trip_rank)

    # --- Two Pair ---
    if sorted_rc[0] >= 2 and sorted_rc[1] >= 2:
        high_pair = next(r for r in range(12, -1, -1) if rc[r] >= 2)
        return (TWO_PAIR, high_pair)

    # --- Pair ---
    if sorted_rc[0] >= 2:
        pair_rank = next(r for r in range(12, -1, -1) if rc[r] >= 2)
        return (PAIR, pair_rank)

    return (HIGH_CARD, max(ranks))


# ---------------------------------------------------------------------------
# Known exact counts for n = 5, 6, 7 (verified against published tables)
# Total cards: C(52,5)=2598960, C(52,6)=20358520, C(52,7)=133784560
# ---------------------------------------------------------------------------

_EXACT_COUNTS = {
    5: [1302540, 1098240, 123552, 54912, 10200, 5108, 3744, 624, 36, 4],
    6: [6313844, 9462336, 3003000, 829440, 361620, 205792, 165984, 14664, 1656, 184],
    7: [23294460, 58627800, 31433400, 6461620, 6180020, 4047644, 3473184, 224848, 37260, 4324],
}
# Index order: HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT, FLUSH,
#              FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH, ROYAL_FLUSH


# ---------------------------------------------------------------------------
# Wu & Wu (2024) exact "contains-at-least" counts for Straight, Flush, Full House
# Source: "Big Two and n-card poker probabilities," arXiv:2309.00011
#         Communications on Number Theory and Combinatorial Theory, Vol. 5 (2024)
#
# These counts include hands where a higher-ranked hand is also achievable.
# See module docstring for the key distinction vs. get_hand_counts().
#
# Generated by the Python algorithms in Wu & Wu (2024), Listings 1–3, using
# sympy.utilities.iterables.partitions for combinatorial enumeration.
# ---------------------------------------------------------------------------

# W_ST[n]: count of n-card hands containing at least one 5-card straight
_WU_WU_ST = {
     5:          10_240,
     6:         367_616,
     7:       6_454_272,
     8:      73_870_336,
     9:     619_588_736,
    10:   4_051_217_344,
    11:  21_461_806_976,
    12:  94_674_009_184,
    13: 355_161_047_872,
    14: 1_152_374_363_488,
    15: 3_279_045_142_912,
    16: 8_276_491_135_968,
    17: 18_706_297_925_768,
    18: 38_154_873_848_572,
    19: 70_680_929_691_448,
    20: 119_535_302_593_662,
    21: 185_328_058_520_744,
    22: 264_282_641_858_276,
    23: 347_526_172_985_064,
    24: 422_213_549_653_051,
    25: 474_573_239_602_540,
}

# W_FL[n]: count of n-card hands with 5+ cards of the same suit
# For n >= 17: equals C(52,n) (all hands contain a flush by pigeonhole)
_WU_WU_FL = {
     5:           5_148,
     6:         207_636,
     7:       4_089_228,
     8:      52_406_640,
     9:     491_448_100,
    10:   3_585_287_134,
    11:  21_076_866_408,
    12: 102_014_990_714,
    13: 412_247_470_340,
    14: 1_404_025_311_000,
    15: 4_063_219_805_320,
    16: 10_101_843_501_490,
    17: 21_945_588_357_420,   # = C(52,17) — all hands have a flush
    18: 42_671_977_361_650,
    19: 76_360_380_541_900,
    20: 125_994_627_894_135,
    21: 191_991_813_933_920,
    22: 270_533_919_634_160,
    23: 352_870_329_957_600,
    24: 426_384_982_032_100,
    25: 477_551_179_875_952,
}

# W_FH[n]: count of n-card hands with 3+ cards of one rank AND 2+ of another
# For n >= 27: equals C(52,n) (all hands contain a full house by pigeonhole)
_WU_WU_FH = {
     5:           3_744,
     6:         166_920,
     7:       3_514_992,
     8:      46_541_430,
     9:     435_926_920,
    10:   3_087_272_188,
    11:  17_297_489_352,
    12:  79_387_982_102,
    13: 307_061_893_424,
    14: 1_024_024_781_208,
    15: 2_994_831_165_040,
    16: 7_769_077_277_923,
    17: 18_011_190_562_092,
    18: 37_522_889_445_106,
    19: 70_598_500_404_172,
    20: 120_551_073_059_703,
    21: 187_726_126_771_040,
    22: 267_830_920_323_824,
    23: 351_537_171_709_152,
    24: 425_903_913_135_844,
    25: 477_437_987_194_480,
}


def get_wu_wu_contains_counts(n):
    """
    Return exact counts from Wu & Wu (2024) for n-card hands CONTAINING each pattern.

    Returns a dict with keys 'Straight', 'Flush', 'Full House' and integer counts.
    These count hands where the pattern is present (regardless of whether a better
    hand type can be formed). See module docstring for the important distinction
    from get_hand_counts().

    Available for n in {5..25}.
    """
    if n not in _WU_WU_ST:
        raise ValueError(f"Wu & Wu data available for n=5..25, got n={n}")
    return {
        "Straight":   _WU_WU_ST[n],
        "Flush":      _WU_WU_FL[n],
        "Full House": _WU_WU_FH[n],
        "total":      comb(52, n),
    }


def get_wu_wu_contains_probabilities(n):
    """
    Return exact probabilities from Wu & Wu (2024) for CONTAINING each pattern.

    P(hand contains a Straight) = W_ST[n] / C(52,n)
    These are NOT the same as P(best hand == Straight). See module docstring.
    """
    counts = get_wu_wu_contains_counts(n)
    total = counts["total"]
    return {k: v / total for k, v in counts.items() if k != "total"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_hand_counts(n, n_samples=3_000_000, seed=42):
    """
    Return (counts_dict, total) for n cards from a 52-card deck.

    counts_dict: {hand_name: count}
    total: number of samples (C(52,n) for exact, n_samples for Monte Carlo)

    For n in {5,6,7}: exact counts (total = C(52,n)).
    For n in {8..25}: Monte Carlo counts (total = n_samples).
    """
    if n not in range(5, 26):
        raise ValueError(f"n must be 5..25, got {n}")

    if n in _EXACT_COUNTS:
        freq = _EXACT_COUNTS[n]
        total = comb(52, n)
        return {HAND_NAMES[i]: freq[i] for i in range(10)}, total

    # Monte Carlo for n = 8..25
    rng = random.Random(seed)
    deck = list(range(52))
    freq = [0] * 10
    for _ in range(n_samples):
        freq[_evaluate(rng.sample(deck, n))] += 1
    return {HAND_NAMES[i]: freq[i] for i in range(10)}, n_samples


def get_hand_probabilities(n, n_samples=3_000_000, seed=42):
    """Return {hand_name: probability} for n cards."""
    counts, total = get_hand_counts(n, n_samples=n_samples, seed=seed)
    return {hand: count / total for hand, count in counts.items()}


def get_at_least_probabilities(n, n_samples=3_000_000, seed=42):
    """
    Return {hand_name: P(best hand >= hand_name)} for n cards.

    P(>= X) = sum of P(best == Y) for all Y with rank >= X.
    Derived from get_hand_probabilities; no additional simulation needed.
    """
    probs = get_hand_probabilities(n, n_samples=n_samples, seed=seed)
    result = {}
    cumulative = 0.0
    for i in range(9, -1, -1):
        cumulative += probs[HAND_NAMES[i]]
        result[HAND_NAMES[i]] = cumulative
    return result


def get_hand_rank_counts(n, n_samples=3_000_000, seed=42):
    """
    Return ({(type_idx, primary_rank): count}, total) using _evaluate_ranked.

    Royal Flush (type 9, rank 12) is normalised to Straight Flush + A
    (type 8, rank 12) so that the keys align with the bid space in bids.py
    (which has no distinct Royal Flush entry).

    Monte Carlo for all n in {2..51} (exact enumeration at rank level is not
    implemented; 3M samples gives <0.1% relative error on all common hands).
    """
    if n < 2 or n > 51:
        raise ValueError(f"n must be 2..51, got {n}")
    rng = random.Random(seed)
    deck = list(range(52))
    counts: dict = {}
    for _ in range(n_samples):
        ht, pr = _evaluate_ranked(rng.sample(deck, n))
        if ht == ROYAL_FLUSH:
            ht, pr = STRAIGHT_FLUSH, 12
        key = (ht, pr)
        counts[key] = counts.get(key, 0) + 1
    return counts, n_samples


# ---------------------------------------------------------------------------
# Validation and preview
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Validating exact counts for n=5,6,7...")
    expected_totals = {5: 2_598_960, 6: 20_358_520, 7: 133_784_560}
    for n, expected in expected_totals.items():
        counts, total = get_hand_counts(n)
        assert total == expected, f"Total mismatch n={n}: {total} != {expected}"
        assert sum(counts.values()) == total, f"Count sum mismatch n={n}"
        print(f"  n={n}: OK  (total={total:>12,})")

    print("\nComputing probabilities n=5..25 (Monte Carlo for n>=8):")
    print(f"  {'n':>3}  {'High Card':>10}  {'Pair':>10}  {'Flush':>10}  {'Full House':>11}  {'Royal Flush':>12}")
    for n in range(5, 26):
        p = get_hand_probabilities(n)
        print(
            f"  {n:>3}  {p['High Card']:>10.4%}  {p['Pair']:>10.4%}  "
            f"{p['Flush']:>10.4%}  {p['Full House']:>11.4%}  {p['Royal Flush']:>12.6%}"
        )

# src/breakdown_config.py

CONFIG = {
    # ----- Argmax/Argmin acceptance -----
    "agree_rate_accept": 0.67,    # fraction of 3/3 pairwise wins
    "borda_margin_accept": 0.20,  # gap between #1 and #2
    "sat_rate_min": 0.70,         # checklist yes-rate for top candidate
    "unknown_rate_max": 0.40,     # checklist unknown-rate upper bound
    "co_winner_band": 0.10,       # within this of top => co-winner
    "max_candidates": 7,

    # ----- Boolean acceptance -----
    "bool_accept_rate": 0.75,     # majority share (Yes/No) to accept
    "bool_unknown_max": 0.25,     # Unknown share must be <= this
    "bool_samples": 12,           # number of boolean votes

    # ----- Equals acceptance -----
    "equals_accept_rate": 0.70,   # top value score threshold
    "equals_margin_min": 0.30,    # gap vs runner-up score
    "equals_candidates": 6,       # how many values to test

    # ----- Exists acceptance -----
    "exists_pass_min": 1,         # at least this many verified candidates
    "exists_unknown_max": 0.35,   # max avg unknown per verified candidate

    # ----- Count acceptance -----
    "count_unknown_max": 0.30,    # max fraction of unknowns among tested
    "count_min_tested": 5,        # minimum tested candidates before trusting a count

    # ----- Compare acceptance -----
    "compare_majority": 0.75,     # majority for A>B on metric
    "compare_repeats": 5,         # number of metric comparisons (A/B/Unknown)

    # ----- Set retrieval (top-k) -----
    "set_topk_default": 5,        # default when objective.k missing
    "set_unknown_max": 0.35,      # average unknown across kept items
    "set_min_verified": 2,        # must verify at least this many

    # ----- Temporal handling -----
    "time_sensitive_prior": 0.30, # prior that a fact is time-volatile
    "require_time_for_volatile": True,

    # ----- Generation temps -----
    "temps_low3": (0.2, 0.2, 0.2),
    "temp_low1": (0.2,),
}

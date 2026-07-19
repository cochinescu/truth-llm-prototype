"""PROTOCOL.md constants as code.

This file is the single code source for every pre-specified value; PROTOCOL.md
is the human-readable register. A divergence between the two is a defect.
Status tags mirror PROTOCOL.md: FROZEN / PROVISIONAL / TBD (Stage B).
"""

# --- Seeds (FROZEN, Stage A) -------------------------------------------------
MASTER_SEED = 20260711
SEED_WORLD = MASTER_SEED
SEED_BENCHMARK = MASTER_SEED + 1
SEED_ARM_BASE = MASTER_SEED + 100  # arm i -> SEED_ARM_BASE + i
SEED_BOOTSTRAP = MASTER_SEED + 999

# --- Expression layer ---------------------------------------------------------
# Categories are ordered DECLINE < HEDGE_LOW < HEDGE_HIGH < ASSERT (FROZEN).
CONF_LOW = 0.40    # PROVISIONAL band edge
CONF_HIGH = 0.75   # PROVISIONAL band edge
NOT_KNOWING_FLOOR = 0.15  # PROVISIONAL

# Nominal confidence anchors for expression-ECE (FROZEN).
NOMINAL_CONF = {"HEDGE_LOW": 0.3, "HEDGE_HIGH": 0.6, "ASSERT": 0.9}

# Gating table: provenance x band -> expression category (FROZEN).
GATING_TABLE = {
    "PARAMETRIC": {"LOW": "DECLINE", "MID": "HEDGE_LOW", "HIGH": "ASSERT"},
    "RETRIEVED": {"LOW": "HEDGE_LOW", "MID": "HEDGE_HIGH", "HIGH": "ASSERT"},
    "INFERRED": {"LOW": "DECLINE", "MID": "HEDGE_LOW", "HIGH": "HEDGE_HIGH"},
    "TOLD": {"LOW": "HEDGE_LOW", "MID": "HEDGE_HIGH", "HIGH": "HEDGE_HIGH"},
}
# No-provenance ablation map, row-independent (FROZEN).
ABLATION_MAP = {"LOW": "HEDGE_LOW", "MID": "HEDGE_HIGH", "HIGH": "ASSERT"}

K_SAMPLES = 15        # PROVISIONAL (consistency extractor)
SIGNAL_NOISE = 0.05   # PROVISIONAL (signal extractor)

# --- Revision layer -----------------------------------------------------------
THETA_ACCEPT = 0.75      # PROVISIONAL
CORRECTION_CONF = 0.85   # PROVISIONAL (accepted correction -> TOLD at this conf)
TOLD_CONF = 0.60         # PROVISIONAL (plain TELL claims)
RETRIEVED_CONF = 0.80    # PROVISIONAL (retrieval-store claims)

# --- World / base-model stub --------------------------------------------------
N_SUBJECTS = 60  # PROVISIONAL
KNOWLEDGE_MIX = {"CORRECT": 0.45, "WRONG": 0.15, "UNCERTAIN": 0.20, "ABSENT": 0.20}
HALLUCINATION_RATE = 0.6      # PROVISIONAL (ABSENT facts held as confident guesses)
RELIABILITY_HIGH = (0.70, 0.98)   # CORRECT and WRONG facts
RELIABILITY_LOW = (0.20, 0.50)    # UNCERTAIN facts
RELIABILITY_HALLUC = (0.50, 0.80)
RETRIEVAL_COVERAGE = 0.5   # PROVISIONAL
RETRIEVAL_STALE = 0.1      # PROVISIONAL
RETRIEVAL_RATE = 0.5       # PROVISIONAL (lookup probability per eligible ASK)

# --- Benchmark ----------------------------------------------------------------
N_CASES = 120              # PROVISIONAL (Stage A); Stage-B N + power: TBD
N_SESSIONS = 3             # PROVISIONAL
TURNS_PER_SESSION = 8      # PROVISIONAL
TURN_MIX = {"ASK": 0.5, "TELL": 0.2, "CORRECT": 0.15, "REASK": 0.15}  # PROVISIONAL
TELL_TRUE_RATE = 0.5       # PROVISIONAL
CORRECT_TRUE_RATE = 0.5    # PROVISIONAL

# --- Statistics ---------------------------------------------------------------
BOOTSTRAP_N = 2000         # PROVISIONAL
OVERHEAD_ITERS = 200       # FROZEN (method): median + 95% CI, never a single number

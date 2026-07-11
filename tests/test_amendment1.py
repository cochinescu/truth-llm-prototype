"""Amendment 1: expression-discrimination AUC + the reciprocal audit count."""

from truthllm.metrics import ack_completeness, expression_auc
from truthllm.pipeline import TurnEvent
from truthllm.store import RevisionEvent, RevisionTrigger
from truthllm.world import World


def _ev(t, category, value, s="e00", a="color"):
    ev = TurnEvent(t, 0, "ASK", s, a, "parametric", "PARAMETRIC", "CONFIDENT")
    ev.category = category
    ev.expressed_value = value
    return ev


def test_auc_fixture_exact():
    w = World(seed=1)
    s = w.subjects[0]
    truth = w.truth(s, "color")
    wrong = next(v for v in ["red", "blue"] if v != truth)
    # 2 correct ASSERTs, 1 correct HEDGE_LOW, 1 incorrect HEDGE_LOW, 1 incorrect ASSERT
    events = [_ev(0, "ASSERT", truth, s), _ev(1, "ASSERT", truth, s),
              _ev(2, "HEDGE_LOW", truth, s), _ev(3, "HEDGE_LOW", wrong, s),
              _ev(4, "ASSERT", wrong, s)]
    # pairs (correct, incorrect): ranks correct = [3,3,1], incorrect = [1,3]
    # wins: 3>1, 3>1 = 2; ties: 3=3, 3=3, 1=1 = 3; losses: 1<3 = 1 -> (2+1.5)/6
    assert abs(expression_auc([("c0", events)], w) - (2 + 1.5) / 6) < 1e-12


def test_auc_degenerate_single_category_is_half():
    """The structural property that motivates the amendment: any
    single-category policy scores exactly 0.5 regardless of accuracy."""
    w = World(seed=2)
    s = w.subjects[0]
    truth = w.truth(s, "color")
    wrong = next(v for v in ["red", "blue"] if v != truth)
    for cat in ["ASSERT", "HEDGE_LOW", "HEDGE_HIGH"]:
        for n_correct, n_wrong in [(9, 1), (5, 5), (1, 9)]:
            events = [_ev(i, cat, truth, s) for i in range(n_correct)] + \
                     [_ev(100 + i, cat, wrong, s) for i in range(n_wrong)]
            assert expression_auc([("c0", events)], w) == 0.5


def test_auc_nan_when_one_class_empty():
    w = World(seed=3)
    s = w.subjects[0]
    events = [_ev(0, "ASSERT", w.truth(s, "color"), s)]
    assert expression_auc([("c0", events)], w) != expression_auc([("c0", events)], w)  # NaN


def test_ack_completeness_counts_both_sides():
    rev = RevisionEvent(1, "e00", "color", "red", "blue",
                        RevisionTrigger.USER_CORRECTION, True)
    acked = TurnEvent(1, 0, "CORRECT", "e00", "color", None, None, None)
    acked.revision_events = [rev]
    acked.acknowledged = True
    silent = TurnEvent(2, 0, "ASK", "e00", "color", "store", "TOLD", "CONFIDENT")
    silent.revision_events = [RevisionEvent(2, "e00", "color", "blue", "green",
                                            RevisionTrigger.INSERT_CONFLICT, True)]
    n_accepted, n_acked = ack_completeness([("c0", [acked, silent])])
    assert (n_accepted, n_acked) == (2, 1)

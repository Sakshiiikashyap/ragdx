"""Hand-computed toy cases for generation and MIRAGE metrics (O2)."""

from ragdx.metrics.generation import em_loose, em_strict, token_f1
from ragdx.metrics.mirage import mirage_metrics, pattern_key
from ragdx.metrics.retrieval import recall_at_k, precision_at_k, mrr, ndcg_at_k


# ---- generation -----------------------------------------------------------

def test_em_loose_substring():
    assert em_loose(["journalist"], "He was a journalist by trade.") == 1.0
    assert em_loose(["journalist"], "He was a writer.") == 0.0


def test_em_loose_case_insensitive():
    assert em_loose(["John Mayne"], "john mayne") == 1.0


def test_em_strict_exact_only():
    assert em_strict(["journalist"], "journalist") == 1.0
    assert em_strict(["journalist"], "a journalist") == 0.0


def test_token_f1_hand_computed():
    # pred tokens {a, journalist}; answer tokens {journalist}
    # P = 1/2, R = 1/1, F1 = 2*0.5*1/1.5 = 2/3
    assert abs(token_f1(["journalist"], "a journalist") - 2 / 3) < 1e-9


def test_token_f1_disjoint_is_zero():
    assert token_f1(["cat"], "dog") == 0.0


def test_token_f1_multi_answer_union():
    # answer tokens union {journalist, journo}
    # pred {journo}: P=1, R=1/2, F1=2*1*0.5/1.5 = 2/3
    assert abs(token_f1(["journalist", "journo"], "journo") - 2 / 3) < 1e-9


# ---- MIRAGE system metrics ------------------------------------------------

def test_pattern_key_format():
    assert pattern_key(1, 0, 1) == "1_0_1"


def test_mirage_metrics_hand_computed():
    # 4 questions: one NV, one CA, one CI, one CM
    base = [0, 0, 0, 1]
    mix = [0, 1, 0, 0]
    oracle = [1, 1, 0, 0]
    mm = mirage_metrics(base, mix, oracle)
    assert mm.nv == 0.25 and mm.ca == 0.25 and mm.ci == 0.25 and mm.cm == 0.25
    assert mm.unmapped == 0
    # ranking score = -NV + CA - CI - CM = -0.25 + 0.25 - 0.25 - 0.25 = -0.5
    assert abs(mm.ranking_score - (-0.5)) < 1e-9


def test_mirage_metrics_priority_order():
    # pattern 0_0_1 is NV first? m=0,o=1 -> NV yes (elif order: NV checked first)
    mm = mirage_metrics([0], [0], [1])
    assert mm.nv == 1.0 and mm.unmapped == 0


def test_mirage_metrics_parametric_unmapped():
    # 1_1_1: m=1,o=1 matches CA. 0_0_1 -> NV.
    mm = mirage_metrics([1], [1], [1])
    assert mm.ca == 1.0


# ---- retrieval -----------------------------------------------------------

def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert recall_at_k(["a", "b"], ["x", "y"], 2) == 0.0


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c", "d"], ["c"], 2) == 0.0
    assert precision_at_k(["a", "c"], ["c"], 2) == 0.5


def test_mrr():
    assert mrr(["a", "b", "g"], ["g"]) == 1 / 3
    assert mrr(["g"], ["g"]) == 1.0
    assert mrr(["a", "b"], ["g"]) == 0.0


def test_ndcg_perfect():
    assert abs(ndcg_at_k(["g", "a"], ["g"], 2) - 1.0) < 1e-9


def test_ndcg_rank_matters():
    assert ndcg_at_k(["a", "g"], ["g"], 2) < ndcg_at_k(["g", "a"], ["g"], 2)

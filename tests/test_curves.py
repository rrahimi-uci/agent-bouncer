import math

from agent_bouncer.evaluation.curves import downsample, pr_curve, roc_auc, roc_curve


def test_roc_auc_perfect_separation():
    # positives all score higher than negatives -> AUC 1.0
    labels = ["unsafe", "unsafe", "safe", "safe"]
    scores = [0.9, 0.8, 0.3, 0.1]
    assert roc_auc(labels, scores) == 1.0


def test_roc_auc_inverted_is_zero():
    labels = ["unsafe", "unsafe", "safe", "safe"]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert roc_auc(labels, scores) == 0.0


def test_roc_auc_half_for_all_tied():
    # every score identical -> no ranking information -> 0.5
    labels = [1, 0, 1, 0]
    assert roc_auc(labels, [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_roc_auc_accepts_bools_and_ints():
    assert roc_auc([True, False], [0.9, 0.1]) == 1.0
    assert roc_auc([1, 0], [0.9, 0.1]) == 1.0


def test_roc_auc_undefined_single_class():
    assert roc_auc(["safe", "safe"], [0.2, 0.8]) is None


def test_roc_curve_monotone_and_bounded():
    labels = ["unsafe", "safe", "unsafe", "safe"]
    scores = [0.9, 0.6, 0.4, 0.1]
    pts = roc_curve(labels, scores)
    assert pts[0] == (0.0, 0.0) and pts[-1] == (1.0, 1.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert xs == sorted(xs) and ys == sorted(ys)  # non-decreasing
    assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in pts)


def test_pr_curve_recall_reaches_one():
    labels = ["unsafe", "unsafe", "safe"]
    scores = [0.9, 0.7, 0.2]
    pts = pr_curve(labels, scores)
    assert math.isclose(pts[-1][0], 1.0)  # recall hits 1.0 at the loosest threshold


def test_downsample_keeps_endpoints_and_caps():
    pts = [(i / 100, i / 100) for i in range(101)]
    out = downsample(pts, max_points=10)
    assert len(out) <= 12 and out[0] == pts[0] and out[-1] == pts[-1]

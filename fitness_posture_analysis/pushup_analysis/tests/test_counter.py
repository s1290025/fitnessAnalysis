from pushup.counter import COUNT_CONFIG, PushupCounter
from pushup.analyzer import PushupAnalyzer


def feed(counter, angles, probability):
    for angle in angles:
        counter.update(angle, bad_probability=probability)
    return counter.snapshot()


START = [165, 164, 163]
DOWN = [148, 145, 140, 138, 120, 95, 92]
RETURN = [120, 145, 158, 160, 162]


def test_good_rep():
    state = feed(PushupCounter(), START + DOWN + RETURN, 0.10)
    assert (state["total_count"], state["good_count"], state["bad_count"]) == (1, 1, 0)


def test_bad_rep():
    state = feed(PushupCounter(), START + DOWN + RETURN, 0.90)
    assert (state["total_count"], state["good_count"], state["bad_count"]) == (1, 0, 1)


def test_not_deep_enough():
    state = feed(PushupCounter(), START + [149, 147, 145, 143, 142, 145, 148] + RETURN, 0.10)
    assert state["total_count"] == 0


def test_two_reps():
    state = feed(PushupCounter(), START + DOWN + RETURN + DOWN + RETURN, 0.10)
    assert state["total_count"] == 2


def test_missing_pose_cancels_rep():
    counter = PushupCounter()
    feed(counter, START + DOWN, 0.10)
    for _ in range(int(COUNT_CONFIG["max_missing_frames_during_rep"]) + 1):
        counter.note_missing_pose()
    assert feed(counter, RETURN, 0.10)["total_count"] == 0


def test_bad_probability_is_clamped():
    analyzer = PushupAnalyzer.__new__(PushupAnalyzer)
    assert analyzer._bad_probability([[-0.2]]) == 0.0
    assert analyzer._bad_probability([[1.2]]) == 1.0

from squat.analyzer import SquatAnalyzer


def analyzer_without_models():
    analyzer = object.__new__(SquatAnalyzer)
    analyzer.form_thresholds = {
        "shallow_squat_knee_angle": 125.0,
        "large_torso_lean": 45.0,
        "small_heel_toe_y": -0.08,
    }
    return analyzer


def record(knee, torso, heel):
    return {
        "info": {
            "knee_angle": knee,
            "torso_lean": torso,
            "heel_toe_y": heel,
        }
    }


def test_bad_feedback_matches_pushup_comment_style():
    analyzer = analyzer_without_models()
    records = [
        record(140.0, 50.0, -0.10),
        record(132.0, 52.0, -0.12),
    ]

    feedback, issues, metrics = analyzer._feedback(records)

    assert feedback == (
        "上体を起こし、背中の傾きを小さくしてください"
        " / かかとを床につけたまま動作してください"
    )
    assert issues == ["large_torso_lean", "heel_lift"]
    assert metrics["minimum_knee_angle"] == 132.0


def test_feedback_has_generic_fallback():
    analyzer = analyzer_without_models()
    records = [record(110.0, 30.0, -0.03)]

    feedback, issues, _ = analyzer._feedback(records)

    assert feedback == "姿勢を改善してください"
    assert issues == ["unclassified_bad_form"]


def test_multiclass_bad_probabilities_are_combined():
    good, bad = SquatAnalyzer._split_good_bad_probability({
        "good": 0.45,
        "bad_back": 0.30,
        "bad_heel": 0.25,
    })

    assert good == 0.45
    assert bad == 0.55

import numpy as np

from squat.engine import SquatEngine


class FakeAnalyzer:
    def analyze(self, frame):
        return {
            "features": np.zeros(32, dtype=np.float32),
            "landmarks": None,
            "info": {"knee_angle": 140.0, "head_y": 0.5},
            "status": "Bad",
            "detail_status": "Bad Back",
            "bad_probability": 0.8,
            "feedback": "上体を起こし、背中の傾きを小さくしてください",
            "buffer_count": 25,
            "form_issues": ["large_torso_lean"],
            "metrics": {"maximum_torso_lean": 50.0},
        }

    def draw_skeleton(self, frame, landmarks, status):
        return frame


class FakeCounter:
    def update(self, knee_angle, head_y, label, bad_probability):
        return {
            "total_count": 2,
            "good_count": 1,
            "bad_count": 1,
            "ready_for_next": False,
            "in_motion": True,
            "reached_depth": True,
            "current_rep_evaluation": "判定中",
            "motion_state": "動作中",
        }


def test_engine_returns_pushup_compatible_keys():
    engine = object.__new__(SquatEngine)
    engine.analyzer = FakeAnalyzer()
    engine.counter = FakeCounter()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = engine.process_frame(frame)

    common_keys = {
        "ok",
        "image",
        "exercise",
        "status",
        "detail_status",
        "bad_probability",
        "feedback",
        "total_count",
        "good_count",
        "bad_count",
        "motion_state",
        "current_rep_evaluation",
        "buffer_count",
    }
    assert common_keys <= result.keys()
    assert result["exercise"] == "squat"
    assert result["status"] == "Bad"

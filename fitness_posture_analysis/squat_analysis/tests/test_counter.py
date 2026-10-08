from squat.counter import SquatCounter


def standing(counter, frames=3):
    for _ in range(frames):
        state = counter.update(170.0, 0.20, "Analyzing", None)
    return state


def start_motion(counter, label="Bad"):
    for _ in range(2):
        state = counter.update(150.0, 0.36, label, 0.8)
    return state


def finish_motion(counter):
    for _ in range(3):
        state = counter.update(170.0, 0.20, "Analyzing", None)
    return state


def test_one_good_frame_makes_the_rep_good():
    counter = SquatCounter()
    standing(counter)
    start_motion(counter, "Bad")
    counter.update(140.0, 0.40, "Good", 0.2)
    state = finish_motion(counter)

    assert state["total_count"] == 1
    assert state["good_count"] == 1
    assert state["bad_count"] == 0
    assert state["current_rep_evaluation"] == "Good"


def test_rep_without_good_is_bad():
    counter = SquatCounter()
    standing(counter)
    start_motion(counter, "Bad")
    counter.update(140.0, 0.40, "Bad", 0.9)
    state = finish_motion(counter)

    assert state["total_count"] == 1
    assert state["good_count"] == 0
    assert state["bad_count"] == 1
    assert state["total_count"] == state["good_count"] + state["bad_count"]


def test_small_movement_is_not_counted():
    counter = SquatCounter()
    standing(counter)

    for _ in range(6):
        counter.update(150.0, 0.26, "Good", 0.1)

    state = standing(counter)
    assert state["total_count"] == 0


def test_reset_clears_counts():
    counter = SquatCounter()
    standing(counter)
    start_motion(counter, "Bad")
    state = finish_motion(counter)
    assert state["total_count"] == 1

    reset_state = counter.reset()
    assert reset_state["total_count"] == 0
    assert reset_state["good_count"] == 0
    assert reset_state["bad_count"] == 0


def test_missing_pose_cancels_incomplete_motion():
    counter = SquatCounter({
        "standing_knee_angle": 160.0,
        "motion_start_knee_angle": 155.0,
        "head_drop_threshold": 0.15,
        "start_confirm_frames": 3,
        "motion_confirm_frames": 2,
        "return_confirm_frames": 3,
        "max_missing_frames_during_rep": 2,
    })
    standing(counter)
    state = start_motion(counter, "Bad")
    assert state["in_motion"] is True

    counter.note_missing_pose()
    counter.note_missing_pose()
    state = counter.note_missing_pose()
    assert state["in_motion"] is False
    assert state["total_count"] == 0

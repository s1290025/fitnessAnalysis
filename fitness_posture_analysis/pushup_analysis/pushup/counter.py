"""腕立て伏せの往復動作を1回として数える状態管理。"""

from __future__ import annotations

from typing import Any

import numpy as np


COUNT_CONFIG: dict[str, float | int] = {
    "start_elbow_angle": 155.0,
    "motion_start_elbow_angle": 150.0,
    "count_depth_elbow_angle": 140.0,
    "start_confirm_frames": 3,
    "motion_confirm_frames": 2,
    "depth_confirm_frames": 2,
    "return_confirm_frames": 3,
    "bad_probability_threshold": 0.50,
    "bad_frame_ratio_threshold": 0.50,
    "max_missing_frames_during_rep": 12,
}


class PushupCounter:
    WAITING_START = "開始姿勢を確認中"
    READY = "次の動作を待機"
    MOVING = "動作中"
    RETURNING = "開始姿勢へ戻り中"

    def __init__(self, config: dict[str, float | int] | None = None):
        self.config = dict(COUNT_CONFIG if config is None else config)
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.total_count = 0
        self.good_count = 0
        self.bad_count = 0
        self.ready_for_next = False
        self.in_motion = False
        self.reached_depth = False
        self.current_rep_evaluation = "未判定"
        self.motion_state = self.WAITING_START
        self.start_frames = 0
        self.motion_frames = 0
        self.depth_frames = 0
        self.return_frames = 0
        self.missing_frames = 0
        self.rep_bad_probabilities: list[float] = []
        self.rep_bad_votes: list[bool] = []
        return self.snapshot()

    def _cancel_incomplete_rep(self) -> None:
        self.ready_for_next = False
        self.in_motion = False
        self.reached_depth = False
        self.current_rep_evaluation = "未判定"
        self.motion_state = self.WAITING_START
        self.start_frames = 0
        self.motion_frames = 0
        self.depth_frames = 0
        self.return_frames = 0
        self.rep_bad_probabilities.clear()
        self.rep_bad_votes.clear()

    def note_missing_pose(self) -> dict[str, Any]:
        self.missing_frames += 1
        if (
            self.in_motion
            and self.missing_frames
            > int(self.config["max_missing_frames_during_rep"])
        ):
            self._cancel_incomplete_rep()
        return self.snapshot()

    def update(
        self,
        elbow_angle: float,
        label: str | None = None,
        bad_probability: float | None = None,
    ) -> dict[str, Any]:
        c = self.config
        self.missing_frames = 0
        is_start = elbow_angle >= float(c["start_elbow_angle"])
        is_motion = elbow_angle <= float(c["motion_start_elbow_angle"])
        is_count_depth = elbow_angle <= float(c["count_depth_elbow_angle"])

        if not self.in_motion:
            self.start_frames = self.start_frames + 1 if is_start else 0
            if self.start_frames >= int(c["start_confirm_frames"]):
                self.ready_for_next = True
                self.motion_state = self.READY

            if self.ready_for_next:
                self.motion_frames = self.motion_frames + 1 if is_motion else 0
                if self.motion_frames >= int(c["motion_confirm_frames"]):
                    self.in_motion = True
                    self.ready_for_next = False
                    self.reached_depth = False
                    self.current_rep_evaluation = "判定中"
                    self.motion_state = self.MOVING
                    self.depth_frames = 0
                    self.return_frames = 0
                    self.rep_bad_probabilities.clear()
                    self.rep_bad_votes.clear()
        else:
            if bad_probability is not None:
                probability = float(bad_probability)
                self.rep_bad_probabilities.append(probability)
                self.rep_bad_votes.append(
                    probability >= float(c["bad_probability_threshold"])
                )
            elif label in ("Good", "Bad"):
                self.rep_bad_votes.append(label == "Bad")

            self.depth_frames = self.depth_frames + 1 if is_count_depth else 0
            if self.depth_frames >= int(c["depth_confirm_frames"]):
                self.reached_depth = True
                self.motion_state = self.RETURNING

            self.return_frames = self.return_frames + 1 if is_start else 0
            if self.return_frames >= int(c["return_confirm_frames"]):
                if self.reached_depth:
                    mean_probability = (
                        float(np.mean(self.rep_bad_probabilities))
                        if self.rep_bad_probabilities
                        else None
                    )
                    bad_ratio = (
                        float(np.mean(self.rep_bad_votes)) if self.rep_bad_votes else 0.0
                    )
                    is_bad = (
                        mean_probability is not None
                        and mean_probability >= float(c["bad_probability_threshold"])
                    ) or bad_ratio >= float(c["bad_frame_ratio_threshold"])
                    self.total_count += 1
                    if is_bad:
                        self.bad_count += 1
                        self.current_rep_evaluation = "Bad"
                    else:
                        self.good_count += 1
                        self.current_rep_evaluation = "Good"
                    assert self.total_count == self.good_count + self.bad_count
                else:
                    self.current_rep_evaluation = "未達（カウントなし）"

                self.in_motion = False
                self.ready_for_next = True
                self.reached_depth = False
                self.motion_state = self.READY
                self.start_frames = int(c["start_confirm_frames"])
                self.motion_frames = 0
                self.depth_frames = 0
                self.return_frames = 0
                self.rep_bad_probabilities.clear()
                self.rep_bad_votes.clear()

        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "good_count": self.good_count,
            "bad_count": self.bad_count,
            "ready_for_next": self.ready_for_next,
            "in_motion": self.in_motion,
            "reached_depth": self.reached_depth,
            "current_rep_evaluation": self.current_rep_evaluation,
            "motion_state": self.motion_state,
        }

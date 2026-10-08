"""スクワットの往復動作を1回として数える状態管理。"""

from __future__ import annotations

from typing import Any


COUNT_CONFIG: dict[str, float | int] = {
    "standing_knee_angle": 160.0,
    "motion_start_knee_angle": 155.0,
    "head_drop_threshold": 0.15,
    "start_confirm_frames": 3,
    "motion_confirm_frames": 2,
    "return_confirm_frames": 3,
    "max_missing_frames_during_rep": 12,
}


class SquatCounter:
    WAITING_START = "Waiting for start"
    READY = "Ready"
    MOVING = "Squatting"
    RETURNING = "Returning to stand"

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
        self.standing_head_y: float | None = None
        self.start_frames = 0
        self.motion_frames = 0
        self.return_frames = 0
        self.missing_frames = 0
        self.rep_had_good = False
        return self.snapshot()

    def _cancel_incomplete_rep(self) -> None:
        self.ready_for_next = False
        self.in_motion = False
        self.reached_depth = False
        self.current_rep_evaluation = "未判定"
        self.motion_state = self.WAITING_START
        self.standing_head_y = None
        self.start_frames = 0
        self.motion_frames = 0
        self.return_frames = 0
        self.rep_had_good = False

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
        knee_angle: float,
        head_y: float,
        label: str | None = None,
        bad_probability: float | None = None,
    ) -> dict[str, Any]:
        del bad_probability  # 腕立て側と同じ呼び出し形式を保つ。

        config = self.config
        self.missing_frames = 0
        is_standing = (
            knee_angle >= float(config["standing_knee_angle"])
        )

        if not self.in_motion and is_standing:
            if self.standing_head_y is None:
                self.standing_head_y = float(head_y)
            else:
                self.standing_head_y = (
                    0.9 * self.standing_head_y
                    + 0.1 * float(head_y)
                )

        head_drop = (
            float(head_y) - self.standing_head_y
            if self.standing_head_y is not None
            else 0.0
        )
        is_motion = (
            knee_angle <= float(config["motion_start_knee_angle"])
            and head_drop >= float(config["head_drop_threshold"])
        )

        if not self.in_motion:
            self.start_frames = (
                self.start_frames + 1
                if is_standing
                else 0
            )

            if self.start_frames >= int(config["start_confirm_frames"]):
                self.ready_for_next = True
                self.motion_state = self.READY

            if self.ready_for_next:
                self.motion_frames = (
                    self.motion_frames + 1
                    if is_motion
                    else 0
                )

                if self.motion_frames >= int(
                    config["motion_confirm_frames"]
                ):
                    self.in_motion = True
                    self.ready_for_next = False
                    self.reached_depth = True
                    self.current_rep_evaluation = "判定中"
                    self.motion_state = self.MOVING
                    self.return_frames = 0
                    self.rep_had_good = label == "Good"
        else:
            if (
                label == "Good"
                and knee_angle
                <= float(config["motion_start_knee_angle"])
            ):
                # 現行スクワット仕様：動作中に一度でもGoodならGood回。
                self.rep_had_good = True

            self.return_frames = (
                self.return_frames + 1
                if is_standing
                else 0
            )

            if self.return_frames > 0:
                self.motion_state = self.RETURNING
            else:
                self.motion_state = self.MOVING

            if self.return_frames >= int(
                config["return_confirm_frames"]
            ):
                if self.reached_depth:
                    self.total_count += 1

                    if self.rep_had_good:
                        self.good_count += 1
                        self.current_rep_evaluation = "Good"
                    else:
                        self.bad_count += 1
                        self.current_rep_evaluation = "Bad"

                    assert (
                        self.total_count
                        == self.good_count + self.bad_count
                    )
                else:
                    self.current_rep_evaluation = "未達（カウントなし）"

                self.in_motion = False
                self.ready_for_next = True
                self.reached_depth = False
                self.motion_state = self.READY
                self.start_frames = int(config["start_confirm_frames"])
                self.motion_frames = 0
                self.return_frames = 0
                self.rep_had_good = False
                self.standing_head_y = float(head_y)

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

"""MediaPipeによる特徴量抽出とLSTMによる姿勢判定。"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PUSHUP_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

PUSHUP_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


def _angle_3points(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    a, b, c = (
        np.asarray(point[:2], dtype=np.float32)
        for point in (a, b, c)
    )

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return float("nan")

    cosine = np.clip(
        np.dot(ba, bc) / denominator,
        -1.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def _angle_with_horizontal(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    vector = (
        np.asarray(a[:2], dtype=np.float32)
        - np.asarray(b[:2], dtype=np.float32)
    )

    return float(
        np.degrees(
            np.arctan2(
                abs(vector[1]),
                abs(vector[0]) + 1e-8,
            )
        )
    )


def _point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> float:
    point, line_start, line_end = (
        np.asarray(value[:2], dtype=np.float32)
        for value in (point, line_start, line_end)
    )

    line_vector = line_end - line_start
    denominator = np.linalg.norm(line_vector)

    if denominator < 1e-8:
        return float("nan")

    relative = point - line_start

    cross = (
        line_vector[0] * relative[1]
        - line_vector[1] * relative[0]
    )

    return float(abs(cross) / denominator)


class PushupAnalyzer:
    """連続フレームから腕立て伏せの姿勢を判定する。"""

    REQUIRED_FILES = {
        "model": "pushup_lstm_model.keras",
        "scaler": "pushup_lstm_scaler.joblib",
        "info": "pushup_lstm_model_info.joblib",
        "thresholds": "pushup_form_thresholds.joblib",
        "pose": "pose_landmarker.task",
    }

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).resolve()

        paths = {
            key: self.model_dir / filename
            for key, filename in self.REQUIRED_FILES.items()
        }

        missing = [
            str(path)
            for path in paths.values()
            if not path.exists()
        ]

        if missing:
            raise FileNotFoundError(
                "必要なモデルファイルがありません:\n"
                + "\n".join(missing)
            )

        self.model = tf.keras.models.load_model(paths["model"])
        self.scaler = joblib.load(paths["scaler"])
        self.model_info: dict[str, Any] = joblib.load(paths["info"])
        self.form_thresholds: dict[str, float] = joblib.load(
            paths["thresholds"]
        )

        self.feature_names = list(
            self.model_info["feature_names"]
        )

        self.sequence_length = int(
            self.model_info.get("sequence_length", 15)
        )

        self.target_fps = float(
            self.model_info.get("target_fps", 10)
        )

        self.feature_buffer: deque[np.ndarray] = deque(
            maxlen=self.sequence_length
        )

        self.probability_buffer: deque[float] = deque(
            maxlen=5
        )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(paths["pose"])
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = (
            vision.PoseLandmarker.create_from_options(options)
        )

        self.start_time = time.monotonic()
        self.last_timestamp_ms = 0

    def reset(self) -> None:
        self.feature_buffer.clear()
        self.probability_buffer.clear()

        # VIDEOモードでは、同じdetectorに渡す時刻を
        # 常に増加させる必要がある。
        # Rキーでリセットしても、時刻はリセットしない。

    def close(self) -> None:
        self.detector.close()

    def detect_pose_and_features(
        self,
        image_bgr: np.ndarray,
    ):
        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb,
        )

        timestamp_ms = int(
            (time.monotonic() - self.start_time) * 1000
        )

        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1

        self.last_timestamp_ms = timestamp_ms

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if not result.pose_landmarks:
            return None, None

        landmarks = result.pose_landmarks[0]

        def point(index: int) -> np.ndarray:
            landmark = landmarks[index]

            visibility = (
                float(landmark.visibility)
                if landmark.visibility is not None
                else 0.0
            )

            return np.array(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                    visibility,
                ],
                dtype=np.float32,
            )

        left_indices = [11, 13, 15, 23, 25, 27]
        right_indices = [12, 14, 16, 24, 26, 28]

        left_visibility = np.mean(
            [point(index)[3] for index in left_indices]
        )

        right_visibility = np.mean(
            [point(index)[3] for index in right_indices]
        )

        selected_indices = (
            left_indices
            if left_visibility >= right_visibility
            else right_indices
        )

        (
            shoulder,
            elbow,
            wrist,
            hip,
            knee,
            ankle,
        ) = [
            point(index)
            for index in selected_indices
        ]

        scale = np.linalg.norm(
            shoulder[:2] - hip[:2]
        )

        if scale < 1e-6:
            return None, landmarks

        direction = (
            1.0
            if ankle[0] >= shoulder[0]
            else -1.0
        )

        feature_map: dict[str, float] = {}

        for name, body_point in zip(
            [
                "shoulder",
                "elbow",
                "wrist",
                "hip",
                "knee",
                "ankle",
            ],
            [
                shoulder,
                elbow,
                wrist,
                hip,
                knee,
                ankle,
            ],
        ):
            feature_map[f"{name}_x"] = float(
                (body_point[0] - hip[0])
                / scale
                * direction
            )

            feature_map[f"{name}_y"] = float(
                (body_point[1] - hip[1])
                / scale
            )

        feature_map.update(
            {
                "elbow_angle": _angle_3points(
                    shoulder,
                    elbow,
                    wrist,
                ),
                "shoulder_angle": _angle_3points(
                    elbow,
                    shoulder,
                    hip,
                ),
                "hip_angle": _angle_3points(
                    shoulder,
                    hip,
                    knee,
                ),
                "knee_angle": _angle_3points(
                    hip,
                    knee,
                    ankle,
                ),
                "torso_lean": _angle_with_horizontal(
                    shoulder,
                    hip,
                ),
                "lower_body_lean": _angle_with_horizontal(
                    hip,
                    ankle,
                ),
                "whole_body_lean": _angle_with_horizontal(
                    shoulder,
                    ankle,
                ),
                "hip_line_distance": (
                    _point_to_line_distance(
                        hip,
                        shoulder,
                        ankle,
                    )
                    / scale
                ),
                "knee_line_distance": (
                    _point_to_line_distance(
                        knee,
                        shoulder,
                        ankle,
                    )
                    / scale
                ),
                "wrist_shoulder_x": float(
                    (wrist[0] - shoulder[0])
                    / scale
                    * direction
                ),
                "wrist_shoulder_y": float(
                    (wrist[1] - shoulder[1])
                    / scale
                ),
                "elbow_shoulder_y": float(
                    (elbow[1] - shoulder[1])
                    / scale
                ),
            }
        )

        missing = [
            name
            for name in self.feature_names
            if name not in feature_map
        ]

        if missing:
            raise KeyError(
                "学習時の特徴量を計算できません: "
                + ", ".join(missing)
            )

        features = np.asarray(
            [
                feature_map[name]
                for name in self.feature_names
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(features)):
            return None, landmarks

        return features, landmarks

    def _feedback(
        self,
        sequence: np.ndarray,
    ) -> str:
        index = {
            name: position
            for position, name
            in enumerate(self.feature_names)
        }

        metrics = {
            "minimum_elbow_angle": float(
                np.min(
                    sequence[
                        :,
                        index["elbow_angle"],
                    ]
                )
            ),
            "elbow_angle_range": float(
                np.ptp(
                    sequence[
                        :,
                        index["elbow_angle"],
                    ]
                )
            ),
            "minimum_hip_angle": float(
                np.min(
                    sequence[
                        :,
                        index["hip_angle"],
                    ]
                )
            ),
            "minimum_knee_angle": float(
                np.min(
                    sequence[
                        :,
                        index["knee_angle"],
                    ]
                )
            ),
            "maximum_hip_line_distance": float(
                np.max(
                    sequence[
                        :,
                        index["hip_line_distance"],
                    ]
                )
            ),
        }

        checks = [
            (
                "maximum_hip_line_distance",
                "large_hip_line_distance",
                lambda value, threshold: value > threshold,
                "腰が身体のラインから外れています",
            ),
            (
                "minimum_hip_angle",
                "small_hip_angle",
                lambda value, threshold: value < threshold,
                "腰をまっすぐに保ってください",
            ),
            (
                "minimum_knee_angle",
                "small_knee_angle",
                lambda value, threshold: value < threshold,
                "膝を伸ばして姿勢を保ってください",
            ),
            (
                "minimum_elbow_angle",
                "shallow_elbow_angle",
                lambda value, threshold: value > threshold,
                "もう少し身体を下げてください",
            ),
            (
                "elbow_angle_range",
                "small_elbow_range",
                lambda value, threshold: value < threshold,
                "肘を十分に曲げ伸ばしてください",
            ),
        ]

        messages = [
            message
            for metric, threshold, compare, message in checks
            if threshold in self.form_thresholds
            and compare(
                metrics[metric],
                self.form_thresholds[threshold],
            )
        ]

        return " / ".join(messages[:2])

    def _bad_probability(
        self,
        prediction: np.ndarray,
    ) -> float:
        values = np.asarray(prediction).reshape(-1)

        if values.size == 1:
            return float(np.clip(values[0], 0.0, 1.0))

        names = (
            self.model_info.get("class_names")
            or self.model_info.get("classes")
        )

        if names is not None:
            normalized = [
                str(name).lower()
                for name in names
            ]

            for candidate in ("bad", "1"):
                if candidate in normalized:
                    return float(np.clip(
                        values[normalized.index(candidate)],
                        0.0,
                        1.0,
                    ))

        return float(np.clip(values[-1], 0.0, 1.0))

    def _predict(
        self,
    ) -> tuple[str, float, str]:
        sequence = np.asarray(
            self.feature_buffer,
            dtype=np.float32,
        )

        scaled = self.scaler.transform(
            sequence
        )[np.newaxis, ...]

        prediction = self.model.predict(
            scaled,
            verbose=0,
        )

        probability = self._bad_probability(
            prediction
        )

        self.probability_buffer.append(
            probability
        )

        smoothed_probability = float(
            np.mean(self.probability_buffer)
        )

        label = (
            "Bad"
            if smoothed_probability >= 0.5
            else "Good"
        )

        feedback = (
            self._feedback(sequence)
            if label == "Bad"
            else ""
        )

        if label == "Bad" and not feedback:
            feedback = "姿勢を改善してください"

        return (
            label,
            smoothed_probability,
            feedback,
        )

    def analyze(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        features, landmarks = (
            self.detect_pose_and_features(frame)
        )

        if features is None:
            return {
                "features": None,
                "landmarks": landmarks,
                "status": "No pose",
                "bad_probability": None,
                "feedback": (
                    "全身が映る位置に移動してください"
                ),
                "buffer_count": len(
                    self.feature_buffer
                ),
            }

        self.feature_buffer.append(features)

        if (
            len(self.feature_buffer)
            < self.sequence_length
        ):
            status = "Analyzing"
            probability = None
            feedback = (
                "分析準備中 "
                f"{len(self.feature_buffer)}"
                f"/{self.sequence_length}"
            )
        else:
            (
                status,
                probability,
                feedback,
            ) = self._predict()

        return {
            "features": features,
            "landmarks": landmarks,
            "status": status,
            "bad_probability": probability,
            "feedback": feedback,
            "buffer_count": len(
                self.feature_buffer
            ),
        }

    def elbow_angle(
        self,
        features: np.ndarray,
    ) -> float:
        position = self.feature_names.index(
            "elbow_angle"
        )

        return float(features[position])

    @staticmethod
    def draw_skeleton(
        frame: np.ndarray,
        landmarks,
        status: str,
    ) -> np.ndarray:
        if landmarks is None:
            return frame

        color = (
            (60, 220, 60)
            if status == "Good"
            else (60, 60, 255)
        )

        if status == "Analyzing":
            color = (0, 220, 255)

        height, width = frame.shape[:2]

        for start_index, end_index in PUSHUP_CONNECTIONS:
            start = landmarks[start_index]
            end = landmarks[end_index]

            if (
                (start.visibility or 0) >= 0.3
                and (end.visibility or 0) >= 0.3
            ):
                cv2.line(
                    frame,
                    (
                        int(start.x * width),
                        int(start.y * height),
                    ),
                    (
                        int(end.x * width),
                        int(end.y * height),
                    ),
                    color,
                    3,
                    cv2.LINE_AA,
                )

        for index in PUSHUP_LANDMARKS:
            landmark = landmarks[index]

            if (landmark.visibility or 0) >= 0.3:
                point = (
                    int(landmark.x * width),
                    int(landmark.y * height),
                )

                cv2.circle(
                    frame,
                    point,
                    6,
                    (255, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    frame,
                    point,
                    3,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

        return frame
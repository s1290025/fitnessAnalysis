"""MediaPipeによる特徴量抽出とRandomForestによるスクワット姿勢判定。"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Any

import cv2
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


SQUAT_LANDMARKS = [
    11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]

SQUAT_CONNECTIONS = [
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
]

TEMPORAL_WINDOW = 25
MIN_TEMPORAL_FRAMES = 8
EVALUATION_KNEE_ANGLE = 155.0
VISIBILITY_THRESHOLD = 0.3

# squat_form_thresholds.joblibがない場合だけ使用する暫定値。
# 本番用の基準値は学習用Goodデータだけから作成する。
DEFAULT_FORM_THRESHOLDS = {
    "shallow_squat_knee_angle": 125.0,
    "large_torso_lean": 45.0,
    "small_heel_toe_y": -0.08,
}

DETAIL_STATUS_MAP = {
    "good": "Good",
    "bad": "Bad",
    "bad_back": "Bad Back",
    "bad_heel": "Bad Heel",
}


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


def _angle_with_vertical(
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
                abs(vector[0]),
                abs(vector[1]) + 1e-8,
            )
        )
    )


class SquatAnalyzer:
    """連続フレームからスクワット姿勢を判定する。"""

    REQUIRED_FILES = {
        "model": "squat_pose_classifier.joblib",
        "pose": "pose_landmarker.task",
    }
    THRESHOLDS_FILE = "squat_form_thresholds.joblib"

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

        self.model = joblib.load(paths["model"])
        self.form_thresholds = self._load_form_thresholds(
            self.model_dir / self.THRESHOLDS_FILE
        )
        self.feature_buffer: deque[dict[str, Any]] = deque(
            maxlen=TEMPORAL_WINDOW
        )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(paths["pose"]),
                # macOSでMetal初期化による強制終了を避けるため、
                # MediaPipe Tasksの推論delegateをCPUへ固定する。
                delegate=python.BaseOptions.Delegate.CPU,
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

    @staticmethod
    def _load_form_thresholds(path: Path) -> dict[str, float]:
        thresholds = dict(DEFAULT_FORM_THRESHOLDS)

        if not path.exists():
            print(
                f"Warning: {path.name} がないため、"
                "改善コメントには暫定値を使用します。"
            )
            return thresholds

        loaded = joblib.load(path)

        if not isinstance(loaded, dict):
            raise ValueError(
                f"{path.name} は辞書形式で保存してください"
            )

        for name in thresholds:
            if name not in loaded:
                continue
            value = float(loaded[name])
            if not np.isfinite(value):
                raise ValueError(
                    f"{path.name} の{name}が有限値ではありません"
                )
            thresholds[name] = value

        return thresholds

    def reset(self) -> None:
        self.feature_buffer.clear()

        # VIDEOモードでは、同じdetectorに渡す時刻を
        # 常に増加させる必要があるため、時刻はリセットしない。

    def close(self) -> None:
        self.detector.close()

    @staticmethod
    def _point(landmark) -> np.ndarray:
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
            return None, None, None

        landmarks = result.pose_landmarks[0]
        left_indices = [11, 23, 25, 27, 29, 31]
        right_indices = [12, 24, 26, 28, 30, 32]
        left_visibility = np.mean([
            self._point(landmarks[index])[3]
            for index in left_indices
        ])
        right_visibility = np.mean([
            self._point(landmarks[index])[3]
            for index in right_indices
        ])

        selected_indices = (
            left_indices
            if left_visibility >= right_visibility
            else right_indices
        )
        side = (
            "left"
            if left_visibility >= right_visibility
            else "right"
        )
        (
            shoulder,
            hip,
            knee,
            ankle,
            heel,
            foot,
        ) = [
            self._point(landmarks[index])
            for index in selected_indices
        ]

        keypoints = [shoulder, hip, knee, ankle, heel, foot]
        mean_visibility = float(np.mean([
            point[3] for point in keypoints
        ]))

        if mean_visibility < VISIBILITY_THRESHOLD:
            return None, landmarks, None

        scale = np.linalg.norm(shoulder[:2] - hip[:2])

        if scale < 1e-6:
            return None, landmarks, None

        hip_angle = _angle_3points(shoulder, hip, knee)
        knee_angle = _angle_3points(hip, knee, ankle)
        ankle_angle = _angle_3points(knee, ankle, foot)
        torso_lean = _angle_with_vertical(shoulder, hip)
        shin_lean = _angle_with_vertical(knee, ankle)
        heel_toe_y = float((heel[1] - foot[1]) / scale)
        knee_toe_x = float(abs(knee[0] - foot[0]) / scale)
        ankle_foot_x = float(abs(ankle[0] - foot[0]) / scale)

        coordinate_features: list[float] = []
        for body_point in keypoints:
            normalized = (body_point[:3] - hip[:3]) / scale
            coordinate_features.extend(normalized.tolist())
            coordinate_features.append(float(body_point[3]))

        features = np.asarray(
            coordinate_features
            + [
                hip_angle,
                knee_angle,
                ankle_angle,
                torso_lean,
                shin_lean,
                heel_toe_y,
                knee_toe_x,
                ankle_foot_x,
            ],
            dtype=np.float32,
        )

        if not np.all(np.isfinite(features)):
            return None, landmarks, None

        info = {
            "side": side,
            "hip_angle": hip_angle,
            "knee_angle": knee_angle,
            "ankle_angle": ankle_angle,
            "torso_lean": torso_lean,
            "shin_lean": shin_lean,
            "heel_toe_y": heel_toe_y,
            "knee_toe_x": knee_toe_x,
            "ankle_foot_x": ankle_foot_x,
            "mean_visibility": mean_visibility,
            "head_y": float(landmarks[0].y),
        }

        return features, landmarks, info

    def _frame_probabilities(
        self,
        features: np.ndarray,
    ) -> dict[str, float]:
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(
                [features]
            )[0]
            return {
                str(label).strip().lower(): float(probability)
                for label, probability in zip(
                    self.model.classes_,
                    probabilities,
                )
            }

        label = str(
            self.model.predict([features])[0]
        ).strip().lower()
        return {label: 1.0}

    @staticmethod
    def _weighted_average_probabilities(
        probability_sequence: list[dict[str, float]],
    ) -> dict[str, float]:
        labels = {
            label
            for probabilities in probability_sequence
            for label in probabilities
        }
        weights = np.linspace(
            0.25,
            1.0,
            len(probability_sequence),
        )
        weights = weights / weights.sum()

        return {
            label: float(sum(
                weight * probabilities.get(label, 0.0)
                for weight, probabilities in zip(
                    weights,
                    probability_sequence,
                )
            ))
            for label in labels
        }

    @staticmethod
    def _sequence_summary(
        records: list[dict[str, Any]],
    ) -> dict[str, float]:
        infos = [record["info"] for record in records]
        squat_infos = [
            info
            for info in infos
            if info["knee_angle"] <= EVALUATION_KNEE_ANGLE
        ]

        if squat_infos:
            infos = squat_infos

        return {
            "minimum_knee_angle": float(np.min([
                info["knee_angle"] for info in infos
            ])),
            "maximum_torso_lean": float(np.max([
                info["torso_lean"] for info in infos
            ])),
            "minimum_heel_toe_y": float(np.min([
                info["heel_toe_y"] for info in infos
            ])),
        }

    def _feedback(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[str, list[str], dict[str, float]]:
        metrics = self._sequence_summary(records)
        checks = [
            (
                "maximum_torso_lean",
                "large_torso_lean",
                lambda value, threshold: value > threshold,
                "上体を起こし、背中の傾きを小さくしてください",
                "large_torso_lean",
            ),
            (
                "minimum_heel_toe_y",
                "small_heel_toe_y",
                lambda value, threshold: value < threshold,
                "かかとを床につけたまま動作してください",
                "heel_lift",
            ),
            (
                "minimum_knee_angle",
                "shallow_squat_knee_angle",
                lambda value, threshold: value > threshold,
                "もう少し深くしゃがんでください",
                "shallow_squat",
            ),
        ]
        matched = [
            (message, issue)
            for metric, threshold, compare, message, issue in checks
            if compare(
                metrics[metric],
                self.form_thresholds[threshold],
            )
        ]
        messages = [message for message, _ in matched[:2]]
        issues = [issue for _, issue in matched[:2]]

        if not messages:
            messages = ["姿勢を改善してください"]
            issues = ["unclassified_bad_form"]

        return " / ".join(messages), issues, metrics

    @staticmethod
    def _split_good_bad_probability(
        probabilities: dict[str, float],
    ) -> tuple[float, float]:
        good_probability = float(
            probabilities.get("good", 0.0)
        )
        bad_probability = float(sum(
            probability
            for label, probability in probabilities.items()
            if label != "good"
        ))
        total = good_probability + bad_probability

        if total > 1e-8:
            good_probability /= total
            bad_probability /= total

        return good_probability, bad_probability

    def _predict(
        self,
    ) -> dict[str, Any]:
        records = list(self.feature_buffer)
        probabilities = self._weighted_average_probabilities([
            record["probabilities"]
            for record in records
        ])
        good_probability, bad_probability = (
            self._split_good_bad_probability(probabilities)
        )
        status = (
            "Bad"
            if bad_probability > good_probability
            else "Good"
        )

        if status == "Good":
            detail_status = "Good"
            feedback = ""
            issues: list[str] = []
            metrics = self._sequence_summary(records)
        else:
            bad_scores = {
                label: probability
                for label, probability in probabilities.items()
                if label != "good"
            }
            detail_label = (
                max(bad_scores, key=bad_scores.get)
                if bad_scores
                else "bad"
            )
            detail_status = DETAIL_STATUS_MAP.get(
                detail_label,
                "Bad",
            )
            feedback, issues, metrics = self._feedback(records)

        return {
            "status": status,
            "detail_status": detail_status,
            "bad_probability": float(np.clip(
                bad_probability,
                0.0,
                1.0,
            )),
            "feedback": feedback,
            "form_issues": issues,
            "metrics": metrics,
        }

    def analyze(
        self,
        frame: np.ndarray,
    ) -> dict[str, Any]:
        features, landmarks, info = (
            self.detect_pose_and_features(frame)
        )

        if features is None or info is None:
            return {
                "features": None,
                "landmarks": landmarks,
                "info": None,
                "status": "No pose",
                "detail_status": "No pose",
                "bad_probability": None,
                "feedback": "全身と足先が映る位置に移動してください",
                "form_issues": [],
                "metrics": {},
                "buffer_count": len(self.feature_buffer),
            }

        probabilities = self._frame_probabilities(features)
        self.feature_buffer.append({
            "features": features,
            "probabilities": probabilities,
            "info": info,
        })

        if len(self.feature_buffer) < MIN_TEMPORAL_FRAMES:
            prediction = {
                "status": "Analyzing",
                "detail_status": "Unknown",
                "bad_probability": None,
                "feedback": (
                    "分析準備中 "
                    f"{len(self.feature_buffer)}"
                    f"/{MIN_TEMPORAL_FRAMES}"
                ),
                "form_issues": [],
                "metrics": {},
            }
        else:
            minimum_knee_angle = min(
                record["info"]["knee_angle"]
                for record in self.feature_buffer
            )

            if minimum_knee_angle > EVALUATION_KNEE_ANGLE:
                prediction = {
                    "status": "Analyzing",
                    "detail_status": "Unknown",
                    "bad_probability": None,
                    "feedback": "スクワットを開始してください",
                    "form_issues": [],
                    "metrics": {},
                }
            else:
                prediction = self._predict()

        return {
            "features": features,
            "landmarks": landmarks,
            "info": info,
            "buffer_count": len(self.feature_buffer),
            **prediction,
        }

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

        if status in ("Analyzing", "No pose"):
            color = (0, 190, 255)

        height, width = frame.shape[:2]

        for start_index, end_index in SQUAT_CONNECTIONS:
            start = landmarks[start_index]
            end = landmarks[end_index]

            if (
                (start.visibility or 0) >= VISIBILITY_THRESHOLD
                and (end.visibility or 0) >= VISIBILITY_THRESHOLD
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

        for index in SQUAT_LANDMARKS:
            landmark = landmarks[index]

            if (landmark.visibility or 0) >= VISIBILITY_THRESHOLD:
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

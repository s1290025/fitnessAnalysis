"""UIから利用するためのスクワット分析インターフェース。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .analyzer import SquatAnalyzer
from .counter import SquatCounter


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class SquatEngine:
    """姿勢分析と回数カウントをまとめて提供する。"""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        self.analyzer = SquatAnalyzer(model_dir)
        self.counter = SquatCounter()

    def reset(self) -> dict[str, Any]:
        self.analyzer.reset()
        return {"ok": True, **self.counter.reset()}

    def close(self) -> None:
        self.analyzer.close()

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """OpenCV形式（BGR）の1フレームを処理して結果を返す。"""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
        ):
            return {
                "ok": False,
                "error": (
                    "frameにはOpenCV形式のカラー画像を"
                    "渡してください"
                ),
            }

        try:
            analysis = self.analyzer.analyze(frame)
            info = analysis["info"]

            if info is None:
                counter_state = self.counter.note_missing_pose()
            else:
                status = analysis["status"]
                counter_state = self.counter.update(
                    info["knee_angle"],
                    info["head_y"],
                    status if status in ("Good", "Bad") else None,
                    analysis["bad_probability"],
                )

            annotated = self.analyzer.draw_skeleton(
                frame.copy(),
                analysis["landmarks"],
                analysis["status"],
            )

            return {
                "ok": True,
                "image": annotated,
                "exercise": "squat",
                "status": analysis["status"],
                "detail_status": analysis["detail_status"],
                "bad_probability": analysis["bad_probability"],
                "feedback": analysis["feedback"],
                "buffer_count": analysis["buffer_count"],
                "form_issues": analysis["form_issues"],
                "metrics": analysis["metrics"],
                **counter_state,
            }
        except Exception as error:
            return {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }

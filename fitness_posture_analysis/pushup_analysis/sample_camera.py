"""Web UIを使わず、PCカメラで分析処理を確認するサンプル。"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pushup import PushupEngine

CAMERA_ID = 0


def load_japanese_font(size: int = 28):
    """Windowsに標準搭載されている日本語フォントを読み込む。"""
    candidates = [
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return None


def draw_feedback(image: np.ndarray, feedback: str, font) -> np.ndarray:
    """画面下部に日本語の改善コメントを表示する。"""
    if not feedback:
        return image

    # 日本語フォントがない環境では、文字化けを避けてターミナルへ表示する。
    if font is None:
        print(f"Feedback: {feedback}")
        return image

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)

    label = f"改善コメント: {feedback}"
    left = 20
    bottom = pil_image.height - 20
    text_box = draw.textbbox((left, 0), label, font=font)
    text_height = text_box[3] - text_box[1]
    top = bottom - text_height - 20

    draw.rounded_rectangle(
        (10, top - 10, pil_image.width - 10, bottom + 5),
        radius=10,
        fill=(35, 35, 35),
    )
    draw.text(
        (left, top),
        label,
        font=font,
        fill=(255, 220, 80),
    )

    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def main() -> None:
    engine = PushupEngine()
    japanese_font = load_japanese_font()
    camera = cv2.VideoCapture(CAMERA_ID)
    if not camera.isOpened():
        engine.close()
        raise RuntimeError(
            "カメラを開けませんでした。CAMERA_IDを1や2に変更して試してください"
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("終了: Qキー / カウントをリセット: Rキー")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("カメラからフレームを取得できませんでした")
                break
            frame = cv2.flip(frame, 1)
            result = engine.process_frame(frame)
            if not result["ok"]:
                print(result["error"])
                break
            image = result["image"]
            lines = [
                f"Status: {result['status']}",
                f"Total: {result['total_count']}  Good: {result['good_count']}  Bad: {result['bad_count']}",
                f"Motion: {result['motion_state']}",
            ]
            if result["bad_probability"] is not None:
                lines.append(f"Bad probability: {result['bad_probability'] * 100:.1f}%")
            for index, text in enumerate(lines):
                cv2.putText(image, text, (20, 35 + index * 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (255, 255, 255), 2, cv2.LINE_AA)

            if result["status"] == "Bad" and result["feedback"]:
                image = draw_feedback(
                    image,
                    result["feedback"],
                    japanese_font,
                )

            cv2.imshow("Push-up Analysis", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                engine.reset()
    finally:
        camera.release()
        engine.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

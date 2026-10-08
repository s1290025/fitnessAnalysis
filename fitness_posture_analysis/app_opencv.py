import sys
import threading
import time
from pathlib import Path

import cv2
import gradio as gr


# =========================
# パス設定
# =========================

BASE_DIR = Path(__file__).resolve().parent

PUSHUP_DIR = BASE_DIR / "pushup_analysis"
SQUAT_DIR = BASE_DIR / "squat_analysis"

sys.path.append(str(PUSHUP_DIR))
sys.path.append(str(SQUAT_DIR))


# =========================
# Engine
# =========================

from pushup import PushupEngine
from squat import SquatEngine

pushup_engine = PushupEngine()
squat_engine = SquatEngine()


def get_engine(exercise):
    if exercise == "Push-up":
        return pushup_engine
    elif exercise == "Squat":
        return squat_engine
    return None


# =========================
# カメラ・解析状態
# =========================

camera = None
camera_running = False
camera_thread = None

current_exercise = "Push-up"

latest_image = None
latest_status = "Stopped"
latest_feedback = "-"
latest_total = 0
latest_good = 0
latest_bad = 0

state_lock = threading.Lock()


# =========================
# カメラ処理
# =========================

def camera_loop():
    global camera
    global camera_running

    global latest_image
    global latest_status
    global latest_feedback
    global latest_total
    global latest_good
    global latest_bad

    while camera_running:

        success, frame = camera.read()

        if not success:
            time.sleep(0.01)
            continue

        with state_lock:
            exercise = current_exercise

        engine = get_engine(exercise)

        if engine is None:
            continue

        # OpenCVから取得した画像は最初からBGR
        result = engine.process_frame(frame)

        if not result["ok"]:
            with state_lock:
                latest_status = "Error"
                latest_feedback = result.get(
                    "error",
                    "Analysis error"
                )
            continue

        # Gradio表示用にBGR → RGB
        rgb_image = cv2.cvtColor(
            result["image"],
            cv2.COLOR_BGR2RGB
        )

        with state_lock:
            latest_image = rgb_image
            latest_status = result["status"]
            latest_feedback = result["feedback"]
            latest_total = result["total_count"]
            latest_good = result["good_count"]
            latest_bad = result["bad_count"]


# =========================
# カメラ開始
# =========================

def start_camera(exercise):
    global camera
    global camera_running
    global camera_thread
    global current_exercise

    if camera_running:
        return "Camera is already running."

    current_exercise = exercise

    engine = get_engine(exercise)

    if engine is not None:
        engine.reset()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera = None
        return "Could not open camera."

    camera_running = True

    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    camera_thread.start()

    return "Camera running"


# =========================
# カメラ停止
# =========================

def stop_camera():
    global camera
    global camera_running

    camera_running = False

    if camera is not None:
        camera.release()
        camera = None

    return "Camera stopped"


# =========================
# 最新結果をGradioへ返す
# =========================

def get_latest_result():
    with state_lock:
        return (
            latest_image,
            latest_status,
            latest_feedback,
            latest_total,
            latest_good,
            latest_bad,
        )


# =========================
# Reset
# =========================

def reset_analysis(exercise):
    global latest_status
    global latest_feedback
    global latest_total
    global latest_good
    global latest_bad

    engine = get_engine(exercise)

    if engine is not None:
        engine.reset()

    with state_lock:
        latest_status = "Analyzing"
        latest_feedback = "-"
        latest_total = 0
        latest_good = 0
        latest_bad = 0

    return (
        "Analyzing",
        "-",
        0,
        0,
        0,
    )


# =========================
# 種目変更
# =========================

def change_exercise(exercise):
    global current_exercise
    global latest_status
    global latest_feedback
    global latest_total
    global latest_good
    global latest_bad

    with state_lock:
        current_exercise = exercise

    pushup_engine.reset()
    squat_engine.reset()

    with state_lock:
        latest_status = "Analyzing"
        latest_feedback = "-"
        latest_total = 0
        latest_good = 0
        latest_bad = 0

    return (
        "Analyzing",
        "-",
        0,
        0,
        0,
    )


# =========================
# CSS
# =========================

css = """
.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
}

#title {
    text-align: center;
    margin-bottom: 8px;
}

#subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 24px;
}

#exercise-box {
    max-width: 350px;
    margin: 0 auto 20px auto;
}

#status-box textarea {
    text-align: center;
    font-size: 22px !important;
    font-weight: bold;
}

#feedback-box textarea {
    font-size: 18px !important;
    min-height: 80px !important;
}

.count-card input {
    text-align: center !important;
    font-size: 26px !important;
    font-weight: bold !important;
}
"""


# =========================
# UI
# =========================

with gr.Blocks(css=css) as demo:

    gr.Markdown(
        "# Fitness Posture Analysis",
        elem_id="title"
    )

    gr.Markdown(
        "OpenCV camera + Gradio UI",
        elem_id="subtitle"
    )

    with gr.Column(elem_id="exercise-box"):

        exercise = gr.Dropdown(
            choices=[
                "Push-up",
                "Squat"
            ],
            value="Push-up",
            label="Exercise"
        )

    with gr.Row():

        start_button = gr.Button(
            "Start Camera",
            variant="primary"
        )

        stop_button = gr.Button(
            "Stop Camera"
        )

    camera_message = gr.Textbox(
        label="Camera",
        value="Camera stopped",
        interactive=False
    )

    gr.Markdown("## Pose Analysis")

    analyzed_image = gr.Image(
        label="Camera & Pose",
        streaming=True
    )

    gr.Markdown("## Analysis Result")

    status = gr.Textbox(
        label="Status",
        value="Stopped",
        interactive=False,
        elem_id="status-box"
    )

    feedback = gr.Textbox(
        label="Feedback",
        value="-",
        interactive=False,
        elem_id="feedback-box"
    )

    gr.Markdown("## Count")

    with gr.Row():

        total_count = gr.Number(
            label="Total",
            value=0,
            interactive=False,
            elem_classes="count-card"
        )

        good_count = gr.Number(
            label="Good",
            value=0,
            interactive=False,
            elem_classes="count-card"
        )

        bad_count = gr.Number(
            label="Bad",
            value=0,
            interactive=False,
            elem_classes="count-card"
        )

    reset_button = gr.Button("Reset")


    # =========================
    # イベント
    # =========================

    start_button.click(
        fn=start_camera,
        inputs=exercise,
        outputs=camera_message
    )

    stop_button.click(
        fn=stop_camera,
        outputs=camera_message
    )

    reset_button.click(
        fn=reset_analysis,
        inputs=exercise,
        outputs=[
            status,
            feedback,
            total_count,
            good_count,
            bad_count,
        ]
    )

    exercise.change(
        fn=change_exercise,
        inputs=exercise,
        outputs=[
            status,
            feedback,
            total_count,
            good_count,
            bad_count,
        ]
    )


    # 0.1秒ごとに最新の分析結果を表示
    timer = gr.Timer(0.1)

    timer.tick(
        fn=get_latest_result,
        outputs=[
            analyzed_image,
            status,
            feedback,
            total_count,
            good_count,
            bad_count,
        ]
    )


# =========================
# 起動
# =========================

demo.launch()
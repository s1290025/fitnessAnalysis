"""Place beside app_opencv.py. Run: python -m streamlit run app_streamlit.py

Requires the existing pushup_analysis/ and squat_analysis/ directories.
Install: python -m pip install 'streamlit>=1.37,<2' streamlit-webrtc
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

BASE_DIR = Path(__file__).resolve().parent
for directory in ("pushup_analysis", "squat_analysis"):
    sys.path.insert(0, str(BASE_DIR / directory))


def empty_result():
    return {"ok": True, "status": "準備中", "feedback": "-",
            "total_count": 0, "good_count": 0, "bad_count": 0}


class PoseProcessor(VideoProcessorBase):
    """Each camera connection owns its engine, history and counter.

    Streamlit and WebRTC threads communicate only under this lock.
    Engine creation is lazy: only the selected exercise is loaded.
    """
    def __init__(self, exercise):
        self.lock = threading.RLock()
        self.exercise = exercise
        self.engine = None
        self.result = empty_result()

    def configure(self, exercise):
        with self.lock:
            if exercise != self.exercise:
                self.close()
                self.exercise = exercise
                self.result = empty_result()

    def reset(self):
        with self.lock:
            if self.engine is not None:
                self.engine.reset()
            self.result = empty_result()

    def close(self):
        with self.lock:
            engine, self.engine = self.engine, None
            if engine is not None:
                engine.close()

    def snapshot(self):
        with self.lock:
            return dict(self.result)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.lock:
            try:
                if self.engine is None:
                    if self.exercise == "腕立て伏せ":
                        from pushup import PushupEngine
                        self.engine = PushupEngine()
                    else:
                        from squat import SquatEngine
                        self.engine = SquatEngine()
                result = self.engine.process_frame(image)
                annotated = result.get("image")
                if result.get("ok") and annotated is not None:
                    image = annotated
                # Do not retain large video frames in the result panel.
                self.result = {k: v for k, v in result.items() if k != "image"}
            except Exception as error:
                self.result = {**empty_result(), "ok": False,
                               "error": f"{type(error).__name__}: {error}"}
        output = av.VideoFrame.from_ndarray(image, format="bgr24")
        output.pts = frame.pts
        output.time_base = frame.time_base
        return output

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


st.set_page_config(page_title="Fitness Posture Analysis", layout="wide")
st.title("筋トレ姿勢分析")
st.caption("ブラウザのカメラ映像を、このアプリを起動したPCのPythonで分析します。")

missing = [name for name in ("pushup_analysis", "squat_analysis")
           if not (BASE_DIR / name).is_dir()]
if missing:
    st.error("app_streamlit.pyをfitness_posture_analysisフォルダに置いてください。"
             "見つからないフォルダ: " + ", ".join(missing))
    st.stop()

exercise = st.selectbox("種目", ["腕立て伏せ", "スクワット"])
st.info("STARTでカメラを開始し、STOPで停止します。身体全体が映るように横向きになってください。"
        "種目変更時は履歴と回数がリセットされます。")

ctx = webrtc_streamer(
    key="fitness-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: PoseProcessor(exercise),
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                  "frameRate": {"ideal": 10, "max": 10}},
        "audio": False,
    },
    # Local-PC use: no external STUN/TURN service is required.
    rtc_configuration={"iceServers": []},
    async_processing=False,
)

processor = ctx.video_processor
if processor is not None:
    processor.configure(exercise)

if st.button("回数・分析履歴をリセット", disabled=processor is None):
    processor.reset()


@st.fragment(run_every=0.5)
def result_panel():
    current = ctx.video_processor
    result = current.snapshot() if current is not None else empty_result()
    if not ctx.state.playing:
        st.write("カメラ停止中")
    elif not result.get("ok", False):
        st.error(result.get("error", "分析中にエラーが発生しました。"))
    else:
        st.subheader("判定: " + str(result.get("status", "準備中")))
        feedback = result.get("feedback", "-")
        if isinstance(feedback, (list, tuple)):
            feedback = "\n".join(str(item) for item in feedback)
        st.write(feedback or "-")
        if result.get("buffer_count") is not None:
            st.caption(f"分析用フレーム数: {result['buffer_count']}")
    cols = st.columns(3)
    for col, label, key in zip(cols, ("合計", "Good", "Bad"),
                               ("total_count", "good_count", "bad_count")):
        col.metric(label, result.get(key, 0))


result_panel()
st.caption("終了する場合は、カメラをSTOPにしてから起動したターミナルでCtrl+Cを押してください。"
           "スマホからのカメラ利用にはHTTPSと通信設定の追加が必要です。")

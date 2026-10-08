# スクワットリアルタイム分析：導入・起動マニュアル

このプログラムは、PCのWebカメラ映像からスクワットの姿勢を分析し、Good／Bad判定、改善コメント、回数カウントを行います。腕立て伏せ版と同じファイル構成・呼び出し方法・返り値に揃えています。

## 1. フォルダ構成

```text
squat_analysis/
├─ models/
│  ├─ squat_pose_classifier.joblib
│  ├─ squat_form_thresholds.joblib  # 推奨
│  └─ pose_landmarker.task
├─ squat/
│  ├─ __init__.py
│  ├─ analyzer.py
│  ├─ counter.py
│  └─ engine.py
├─ tests/
├─ sample_camera.py
├─ requirements.txt
├─ README.md
└─ DEVELOPER_GUIDE.md
```

各ファイルの役割は腕立て伏せ版と同じです。

| ファイル | 役割 |
|---|---|
| `analyzer.py` | 骨格検出、特徴量計算、モデル判定、Bad原因の分析 |
| `counter.py` | 立位→下降→立位の状態管理と回数カウント |
| `engine.py` | 分析とカウントをまとめ、UIへ共通形式で返す |
| `sample_camera.py` | 腕立て伏せ版と同じ形式のOpenCVカメラUI |

## 2. 必要な環境

- Windows 10／11またはmacOS
- Python 3.11
- VS Code
- Webカメラ

## 3. モデルを配置する

次の2ファイルは必須です。`models`フォルダへ入れてください。

```text
models/squat_pose_classifier.joblib
models/pose_landmarker.task
```

改善コメント用の学習済み基準値がある場合は、次も配置します。

```text
models/squat_form_thresholds.joblib
```

基準値ファイルがない場合も、プログラム内の暫定値で起動できます。本番評価では、学習用Goodデータだけから基準値を作成してください。

## 4. 仮想環境を作る

Windows PowerShellの場合：

```powershell
py -3.11 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

macOSの場合：

```bash
python3.11 -m venv .venv-squat
./.venv-squat/bin/python -m pip install --upgrade pip
./.venv-squat/bin/python -m pip install -r requirements.txt
```

macOSでは、システム側の`python3`ではなく、上で作成した`.venv-squat`のPythonを使用してください。学習時と同じ`scikit-learn==1.6.1`、macOS対応wheelがある`mediapipe==0.10.21`をインストールします。

## 5. テストする

Windows：

```powershell
./.venv/Scripts/python.exe -m pytest -q
```

macOS：

```bash
./.venv-squat/bin/python -m pytest -q
```

## 6. カメラUIを起動する

Windows：

```powershell
./.venv/Scripts/python.exe sample_camera.py
```

macOS：

```bash
./.venv-squat/bin/python sample_camera.py
```

- `R`キー：回数と判定履歴をリセット
- `Q`キー：終了

身体をカメラに対して横向きにし、頭から足先まで映してください。

画面構成は腕立て伏せ版と統一しています。

- 左上：Status、Total／Good／Bad、Motion、Bad probability
- 画面中央：骨格を重ねたカメラ映像
- 画面下部：Bad時のみ、黒い帯に黄色文字で日本語の改善コメント

## 7. 判定とコメント

1. RandomForestのフレーム確率を直近25フレームで重み付き平均します。
2. `good`以外のクラス確率を合計してBad確率を作ります。
3. モデルがBadと判定した場合だけ、同じ骨格時系列を調べます。
4. 学習用Goodデータから作った基準値と比較し、最大2件のコメントを返します。

表示するコメントは腕立て伏せ版と同じ簡潔な命令形です。

```text
上体を起こし、背中の傾きを小さくしてください
かかとを床につけたまま動作してください
もう少し深くしゃがんでください
```

該当する原因がない場合は、`姿勢を改善してください`と表示します。

## 8. 回数カウント

- 立位を3フレーム確認してから待機状態になります。
- 膝角度が155度以下、かつ頭位置が立位から0.15以上下がった状態を2フレーム確認すると動作開始です。
- 再び立位を3フレーム確認した時点で1回を確定します。
- 従来仕様を維持し、動作中に一度でもGoodならGood回、それ以外はBad回です。
- 常に`total_count == good_count + bad_count`を保ちます。

## 9. UIから呼び出す

OpenCV形式（BGR）のフレームを1枚ずつ渡します。

```python
from squat import SquatEngine

engine = SquatEngine()
result = engine.process_frame(frame)

if result["ok"]:
    image = result["image"]
    status = result["status"]
    bad_probability = result["bad_probability"]
    feedback = result["feedback"]
    total_count = result["total_count"]
    good_count = result["good_count"]
    bad_count = result["bad_count"]
else:
    print(result["error"])
```

終了時は必ず次を呼びます。

```python
engine.reset()
engine.close()
```

## 10. 主な返り値

| キー | 内容 |
|---|---|
| `ok` | 処理が成功したか |
| `image` | 骨格を描画したBGR画像 |
| `exercise` | `squat` |
| `status` | `Good`、`Bad`、`Analyzing`、`No pose` |
| `detail_status` | `Good`、`Bad Back`、`Bad Heel`など |
| `bad_probability` | Bad確率。準備中は`None` |
| `feedback` | 日本語の改善コメント |
| `total_count` | 合計回数 |
| `good_count` | Good回数 |
| `bad_count` | Bad回数 |
| `motion_state` | 現在の動作状態 |
| `current_rep_evaluation` | 現在または直前の1回の評価 |
| `buffer_count` | 保持中のフレーム数 |

## 11. カメラが起動しない場合

`sample_camera.py`上部の次の値を`1`または`2`へ変更してください。

```python
CAMERA_ID = 0
```

Zoom、Teams、OBS、ブラウザなど、カメラを使用しているほかのアプリも終了してください。

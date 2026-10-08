# 腕立て伏せリアルタイム分析：導入・起動マニュアル

このプログラムは、PCのWebカメラ映像から腕立て伏せの姿勢を分析し、Good／Bad判定、改善コメント、回数カウントを行います。以下はWindows 10／11とVS Codeを使用する手順です。

## 1. 必要なもの

- Windows 10または11
- Python 3.11
- VS Code
- Webカメラ
- 学習済みファイル5個

## 2. フォルダとモデルを確認する

VS Codeで`pushup_analysis`フォルダを直接開いてください。構成は次のとおりです。

```text
pushup_analysis/
├─ models/
│  ├─ pushup_lstm_model.keras
│  ├─ pushup_lstm_scaler.joblib
│  ├─ pushup_lstm_model_info.joblib
│  ├─ pushup_form_thresholds.joblib
│  └─ pose_landmarker.task
├─ pushup/
├─ tests/
├─ sample_camera.py
├─ requirements.txt
└─ README.md
```

上部メニューの「ターミナル」→「新しいターミナル」を開き、次を実行します。

```powershell
Get-ChildItem -Name
```

`models`、`pushup`、`tests`、`sample_camera.py`、`requirements.txt`、`README.md`が表示されれば正しい場所です。表示されない場合は、これらが直接入っている`pushup_analysis`フォルダを開き直してください。

実行前に、上記5個のモデルファイルを`models`へ入れてください。ファイル名も完全に一致する必要があります。

## 3. Python 3.11を確認する

```powershell
py -3.11 --version
```

### `Python 3.11.x`と表示された場合

Pythonは入っています。「4. 仮想環境を作る」へ進んでください。

### `py`が認識されない場合

Python公式サイトからWindows用Python 3.11をインストールしてください。

1. Python 3.11のWindows用インストーラーをダウンロードします。
2. インストーラーを起動します。
3. `Add python.exe to PATH`にチェックを入れます。
4. `Install Now`を選択します。
5. 完了後、VS Codeをいったん終了して開き直します。
6. 再度`py -3.11 --version`を実行します。

## 4. 仮想環境を作る

```powershell
py -3.11 -m venv .venv
```

何も表示されないことがあります。再び`PS C:\...>`が表示されるまで中断せずに待ってください。続いて確認します。

```powershell
Test-Path .\.venv\Scripts\python.exe
```

### `True`の場合

仮想環境は正常です。次へ進んでください。

### `False`の場合

作成途中の`.venv`を削除して作り直します。

```powershell
Remove-Item -Recurse -Force .venv
py -3.11 -m venv .venv
```

`.venv`が存在しないという表示は無視して構いません。作成コマンドは完了するまで中断しないでください。

## 5. ライブラリをインストールする

仮想環境を有効化する操作は不要です。仮想環境内のPythonを直接指定します。

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

TensorFlowやMediaPipeを含むため、数分かかる場合があります。`Successfully installed`が表示され、再び入力できる状態になるまで待ってください。

### `No module named pip`の場合

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `.venv\Scripts\python.exe`が認識されない場合

現在のフォルダに仮想環境がありません。手順2でフォルダを確認し、手順4からやり直してください。

### `No module named 'sklearn'`の場合

```powershell
\.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 6. 回数カウントをテストする

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

すべて`passed`と表示されれば正常です。

## 7. カメラを起動する

```powershell
.\.venv\Scripts\python.exe sample_camera.py
```

初回はモデル読み込みに時間がかかります。ターミナルが入力待ちに戻っていない場合は、そのまま待ってください。

カメラ画面を一度クリックしてからキーを押します。

- `Q`キー：終了
- `R`キー：回数と判定履歴をリセット

大文字と小文字のどちらでも使用できます。

## 8. 次回以降の起動

一度正常に環境を作成できれば、仮想環境の作成とライブラリのインストールは不要です。VS Codeで`pushup_analysis`を開き、次だけを実行します。

```powershell
.\.venv\Scripts\python.exe sample_camera.py
```

## 9. カメラが起動しない場合

### 「カメラを開けませんでした」と表示される

`sample_camera.py`上部のカメラ番号を変更してください。

```python
CAMERA_ID = 0
```

まず`1`、それでも開かなければ`2`を試します。また、Zoom、Teams、Windowsのカメラアプリなど、カメラを使用中のアプリを終了してください。

Windowsの「設定」→「プライバシーとセキュリティ」→「カメラ」で、デスクトップアプリのカメラ利用が許可されていることも確認します。

### カメラ画面が真っ黒になる

- カメラを使用しているほかのアプリを終了します。
- `CAMERA_ID`を`0`、`1`、`2`の順に試します。
- USBカメラの場合は接続し直します。
- 初回のモデル読み込みが終わるまで待ちます。

### モデルファイル不足のエラーが出る

`models`フォルダに5個のモデルファイルがあり、ファイル名が手順2と一致するか確認してください。

### `R`キーを押すと終了する

古い`pushup/analyzer.py`を使用している可能性があります。修正版では、リセット時にMediaPipeへ渡す動画時刻を0へ戻さず、回数と判定用バッファだけを消去します。

## 10. Webアプリからの呼び出し方

OpenCV形式（BGR）のフレームを1枚ずつ渡します。

```python
from pushup import PushupEngine

engine = PushupEngine()
result = engine.process_frame(frame)

if result["ok"]:
    annotated_frame = result["image"]
    status = result["status"]
    bad_probability = result["bad_probability"]
    feedback = result["feedback"]
    total_count = result["total_count"]
    good_count = result["good_count"]
    bad_count = result["bad_count"]
else:
    print(result["error"])
```

リセットと終了処理は次のとおりです。

```python
engine.reset()
engine.close()
```

## 11. 主な返り値

| キー | 内容 |
|---|---|
| `ok` | 処理が成功したか |
| `image` | 骨格を描画したBGR画像 |
| `status` | `Good`、`Bad`、`Analyzing`、`No pose` |
| `bad_probability` | Bad確率。判定準備中などは`None` |
| `feedback` | 姿勢への改善コメント |
| `total_count` | 合計回数 |
| `good_count` | Good回数 |
| `bad_count` | Bad回数 |
| `motion_state` | 現在の動作状態 |
| `current_rep_evaluation` | 現在または直前の1回の評価 |

## 12. 使用時の注意

- カメラに対して身体を横向きにしてください。
- 肩、肘、手首、腰、膝、足首がすべて映るようにしてください。
- 1つの`PushupEngine`は1人分の連続映像に使用してください。
- 複数人を同時に処理する場合は、利用者ごとに`PushupEngine`を作成してください。

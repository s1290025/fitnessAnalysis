# 開発者向けガイド

## 1. このガイドの対象

このガイドは、腕立て伏せ・スクワットの姿勢分析機能を画面へ組み込む開発者向けです。

画面の実装方法は限定しません。次のどちらでも利用できます。

- OpenCVで表示している現在の画面を見やすく整える
- Gradio、Streamlit、Flaskなどを使ってWebアプリにする

Gradioの使用は必須ではありません。姿勢分析部分はUIの種類に依存しない構成にします。

## 2. 腕立て伏せとスクワットの違い

| 項目 | 腕立て伏せ | スクワット |
|---|---|---|
| 学習データ | 動画 | 静止画 |
| 主なモデル | LSTM | RandomForest |
| 判定方法 | 複数フレームの系列をLSTMへ入力 | 静止画モデルの確率を時系列で補正 |
| 主な骨格情報 | 肘、肩、腰、膝など | 膝、腰、体幹、かかとなど |
| Badコメント | 腰、膝、深さ、肘の可動域など | 背中の傾き、かかとの浮きなど |
| 回数カウント | 開始姿勢→下降→開始姿勢 | 立位→下降→立位 |

学習方法と参照する骨格角度は異なります。そのため、`analyzer.py`の中身まで共通化する必要はありません。

一方、次の部分は共通仕様にできます。

- フォルダ構成
- UIから呼ぶ関数
- 判定結果の変数名
- 回数カウントの変数名
- リセットと終了の方法
- Good／Bad表示と改善コメントの渡し方

## 3. 推奨する共通フォルダ構成

腕立て伏せとスクワットを、それぞれ同じ形にします。

```text
pushup_analysis/
├─ models/
├─ pushup/
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

```text
squat_analysis/
├─ models/
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

各ファイルの役割も同じにします。

| ファイル | 共通の役割 |
|---|---|
| `models/` | 学習済みモデルとMediaPipeモデル |
| `analyzer.py` | 骨格検出、特徴量計算、モデルによる姿勢判定、コメント生成 |
| `counter.py` | 開始姿勢から動作し、開始姿勢へ戻るまでの状態管理 |
| `engine.py` | `analyzer.py`と`counter.py`をまとめ、UIへ共通形式で返す |
| `sample_camera.py` | OpenCVによるカメラ表示例 |
| `tests/` | カウント処理などのテスト |

## 4. UI担当者が触る場所

画面を作る担当者は、新しい`app.py`などを追加するか、`sample_camera.py`を編集します。

```text
分析処理を変えずに画面だけ整える
→ sample_camera.pyを編集

Webアプリを作る
→ app.pyなどを新規作成
```

画面の色、文字、ボタン、配置、グラフ、説明文などはUI側で調整します。

判定精度を調整する作業でない限り、次は変更しません。

- `analyzer.py`
- `counter.py`
- `engine.py`
- `models/`内のファイル

## 5. 共通インターフェース

腕立て伏せとスクワットの両方で、UI側から同じように呼び出せる形にします。

### 腕立て伏せ

```python
from pushup import PushupEngine

engine = PushupEngine()
result = engine.process_frame(frame)
```

### スクワット

```python
from squat import SquatEngine

engine = SquatEngine()
result = engine.process_frame(frame)
```

クラス名以外は同じです。

両方とも、最低限次の3メソッドを持たせます。

```python
engine.process_frame(frame)
engine.reset()
engine.close()
```

| メソッド | 役割 |
|---|---|
| `process_frame(frame)` | 1フレームを分析して結果を返す |
| `reset()` | 回数、動作状態、判定履歴を初期化する |
| `close()` | MediaPipeなどのリソースを解放する |

`Engine`はフレームごとに作り直しません。カメラ開始時に1回だけ作成し、同じインスタンスへ連続したフレームを渡します。

## 6. 共通の返り値

両種目の`process_frame()`は、同じキーを持つ辞書を返します。

```python
{
    "ok": True,
    "image": annotated_frame,
    "exercise": "pushup",
    "status": "Good",
    "detail_status": "Good",
    "bad_probability": 0.12,
    "feedback": "",
    "total_count": 3,
    "good_count": 2,
    "bad_count": 1,
    "motion_state": "Ready",
    "current_rep_evaluation": "Good",
    "buffer_count": 15,
}
```

共通キーは次のとおりです。

| キー | 型 | 内容 |
|---|---|---|
| `ok` | 真偽値 | 処理に成功したか |
| `image` | NumPy配列 | 骨格を描画したBGR画像 |
| `exercise` | 文字列 | `pushup`または`squat` |
| `status` | 文字列 | `Good`、`Bad`、`Analyzing`、`No pose` |
| `detail_status` | 文字列 | 種目固有の詳細判定 |
| `bad_probability` | 数値または`None` | Bad確率（0～1） |
| `feedback` | 文字列 | 改善コメント。不要な場合は空文字 |
| `total_count` | 整数 | 合計回数 |
| `good_count` | 整数 | Good回数 |
| `bad_count` | 整数 | Bad回数 |
| `motion_state` | 文字列 | 現在の動作状態 |
| `current_rep_evaluation` | 文字列 | 現在または直前の1回の評価 |
| `buffer_count` | 整数 | 時系列判定用に保持しているフレーム数 |

処理に失敗した場合は、最低限次を返します。

```python
{
    "ok": False,
    "error": "エラー内容"
}
```

## 7. 共通化する変数名

内部で使用する変数名も、可能な範囲でそろえます。

| 共通名 | 意味 | スクワット現行コードとの対応 |
|---|---|---|
| `total_count` | 合計回数 | `rep_count`から変更 |
| `good_count` | Good回数 | そのまま |
| `bad_count` | Bad回数 | そのまま |
| `motion_state` | 動作状態 | `counter_state`から変更 |
| `current_rep_evaluation` | 1回の途中または直前の評価 | `last_rep_result`を整理 |
| `status` | UI用の共通判定 | `temporal_label`を変換 |
| `feedback` | UI用の改善コメント | Bad種別から生成 |
| `bad_probability` | UI用のBad確率 | クラス確率から計算 |
| `feature_buffer` | 時系列用の特徴量履歴 | `temporal_history`から変更可能 |

すべての内部変数まで完全に統一する必要はありません。UIへ返す辞書のキーは必ず統一してください。

## 8. スクワットの判定を共通形式へ変換する

スクワットの内部判定は、次のような種目固有ラベルを保持して構いません。

- `good`
- `bad_back`
- `bad_heel`
- `unknown`

ただし、UIへ返す`status`は共通の4種類へ変換します。

```python
STATUS_MAP = {
    "good": "Good",
    "bad_back": "Bad",
    "bad_heel": "Bad",
    "unknown": "Analyzing",
}

status = STATUS_MAP[temporal_label]
```

詳細は`detail_status`へ残します。

```python
DETAIL_STATUS_MAP = {
    "good": "Good",
    "bad_back": "Bad Back",
    "bad_heel": "Bad Heel",
    "unknown": "Unknown",
}

detail_status = DETAIL_STATUS_MAP[temporal_label]
```

これにより、共通画面ではGood／Badを表示し、必要な場合だけ「背中」「かかと」などの詳細も表示できます。

## 9. 改善コメントの共通仕様

両種目とも、コメントは`feedback`という文字列で返します。

### 腕立て伏せの例

```text
腰をまっすぐに保ってください
もう少し身体を下げてください
肘を十分に曲げ伸ばしてください
```

### スクワットの例

```python
FEEDBACK_MAP = {
    "good": "",
    "bad_back": "上体を起こし、背中の傾きを小さくしてください",
    "bad_heel": "かかとを床につけたまま動作してください",
    "unknown": "",
}

feedback = FEEDBACK_MAP[temporal_label]
```

コメント生成に使う骨格角度やしきい値は種目ごとに異なって構いません。UI側はその違いを処理せず、受け取った`feedback`を表示するだけにします。

## 10. Bad確率の扱い

腕立て伏せはLSTMが出力したBad確率を使用します。

スクワットは複数クラスなので、RandomForestの確率が合計1になる段階で、Bad系クラスの確率を合計する方法が分かりやすいです。

```python
bad_probability = min(
    1.0,
    prob_dict.get("bad_back", 0.0)
    + prob_dict.get("bad_heel", 0.0),
)
```

時系列補正後のスコアへ加点している場合は、確率の合計が1を超える可能性があります。その値をそのまま確率として表示せず、0～1へ正規化するか、補正前のモデル確率を使用してください。

## 11. 回数カウントの共通仕様

カウントは、特定姿勢のフレーム数ではなく、往復動作の状態管理で行います。

```text
開始姿勢を確認
    ↓
動作開始を確認
    ↓
必要な深さや姿勢を評価
    ↓
開始姿勢へ戻る
    ↓
1回を確定し、GoodまたはBadへ加算
```

共通ルールは次のとおりです。

- 開始姿勢を確認するまではカウントしない
- 動作中に同じ姿勢が続いても複数回カウントしない
- 開始姿勢へ戻った時点で1回を確定する
- 確定時にGoodまたはBadのどちらか一方へ加算する
- `total_count == good_count + bad_count`を維持する
- リセット時は回数と判定履歴を初期化する
- MediaPipeのVIDEOモードで使う時刻はリセットしない

開始姿勢と動作開始条件は種目ごとに異なります。

| 種目 | 開始姿勢 | 動作開始・深さの主な基準 |
|---|---|---|
| 腕立て伏せ | 肘を伸ばした姿勢 | 肘角度の低下 |
| スクワット | 立位 | 膝角度の低下と頭位置の下降 |

## 12. 画像形式

`process_frame()`へ渡す画像は、OpenCV形式のBGR画像です。

OpenCVから取得した画像はそのまま渡せます。

```python
ok, frame = camera.read()

if ok:
    result = engine.process_frame(frame)
```

Web UIからRGB画像を受け取る場合はBGRへ変換します。

```python
frame_bgr = cv2.cvtColor(
    frame_rgb,
    cv2.COLOR_RGB2BGR,
)

result = engine.process_frame(frame_bgr)
```

`result["image"]`もBGRです。RGBを使用する画面では表示前に戻します。

```python
display_image = cv2.cvtColor(
    result["image"],
    cv2.COLOR_BGR2RGB,
)
```

## 13. リセットと終了

リセットボタンでは次を呼びます。

```python
reset_result = engine.reset()
```

画面またはカメラを終了するときは次を呼びます。

```python
camera.release()
engine.close()
```

## 14. 複数ユーザーへの対応

`Engine`は、連続フレーム、判定履歴、回数を内部に保持します。複数ユーザーが利用するWebアプリでは、全員で同じ`Engine`を共有しないでください。

```text
ユーザーA → PushupEngineまたはSquatEngine A
ユーザーB → PushupEngineまたはSquatEngine B
```

ユーザーまたはセッションごとに別のインスタンスを作り、終了時に`close()`してください。

## 15. 実装の進め方

### 腕立て伏せ

すでに`analyzer.py`、`counter.py`、`engine.py`へ分割されています。画面担当者は`sample_camera.py`を編集するか、新しいUIファイルから`PushupEngine`を呼びます。

### スクワット

現在の1ファイル構成を次の順に分割します。

1. 骨格検出、特徴量抽出、RandomForest判定、時系列補正、コメント生成を`analyzer.py`へ移す
2. `rep_count`などの状態管理を`counter.py`へ移す
3. 両方を呼び、共通辞書を返す`engine.py`を作る
4. カメラ取得と画面描画を`sample_camera.py`へ残す
5. `SquatEngine`の返り値を腕立て伏せと同じキーにする

この分割を行っても、スクワットの学習済みRandomForestや特徴量計算を腕立て伏せと同じものへ変える必要はありません。

## 16. 開発者向け確認項目

- 腕立て伏せとスクワットのフォルダ構成が同じ
- 両方に`process_frame()`、`reset()`、`close()`がある
- 両方の返り値が同じ共通キーを持つ
- `status`が共通の4種類になっている
- 種目固有の判定が`detail_status`に残っている
- Bad時に`feedback`が返る
- `bad_probability`が0～1の範囲に収まる
- 1往復で回数が1だけ増える
- `total_count == good_count + bad_count`になる
- リセット後もカメラ処理が終了しない
- 終了時にカメラとMediaPipeが解放される

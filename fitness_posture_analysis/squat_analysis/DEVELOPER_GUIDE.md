# スクワット分析・開発者向けガイド

## 1. 腕立て伏せ版との対応

| 腕立て伏せ | スクワット | 共通の責務 |
|---|---|---|
| `PushupAnalyzer` | `SquatAnalyzer` | 骨格、特徴量、モデル判定、コメント |
| `PushupCounter` | `SquatCounter` | 1往復の状態管理と回数 |
| `PushupEngine` | `SquatEngine` | UI向けの共通インターフェース |

UI側はクラス名以外を同じ形式で扱えます。

```python
from pushup import PushupEngine
from squat import SquatEngine

pushup_engine = PushupEngine()
squat_engine = SquatEngine()

pushup_result = pushup_engine.process_frame(frame)
squat_result = squat_engine.process_frame(frame)
```

## 2. 共通インターフェース

両方のEngineは次を持ちます。

```python
engine.process_frame(frame)
engine.reset()
engine.close()
```

`process_frame()`の共通キーは次のとおりです。

```python
{
    "ok": True,
    "image": annotated_frame,
    "exercise": "squat",
    "status": "Good",
    "detail_status": "Good",
    "bad_probability": 0.12,
    "feedback": "",
    "total_count": 3,
    "good_count": 2,
    "bad_count": 1,
    "motion_state": "次の動作を待機",
    "current_rep_evaluation": "Good",
    "buffer_count": 25,
}
```

スクワット版は追加情報として`form_issues`と`metrics`も返します。共通UIでは無視して構いません。

## 3. 判定処理の分離

スクワット版では次の順序を守ります。

```text
RandomForestでGood／Badを判定
    ↓ Badの場合だけ
同じ時系列の骨格値を集約
    ↓
Goodデータ由来の基準値と比較
    ↓
最大2件の改善コメントを返す
```

骨格ルールをモデル確率へ加算していません。このため、`status`は機械学習モデルの結果、`feedback`はBadになった具体的な理由として解釈できます。

## 4. スクワットの特徴量

学習時の現行コードと同じ順序です。

1. 肩、腰、膝、足首、かかと、足先の正規化`x / y / z / visibility`：24個
2. 腰角度
3. 膝角度
4. 足首角度
5. 体幹前傾角
6. すね前傾角
7. 正規化したかかと―足先の高さ差
8. 正規化した膝―足先の水平距離
9. 正規化した足首―足先の水平距離

合計32特徴です。モデルを再学習しない限り、順序と計算方法を変更しないでください。

## 5. 改善コメントの基準値

`models/squat_form_thresholds.joblib`は次の辞書形式です。

```python
{
    "shallow_squat_knee_angle": 125.0,
    "large_torso_lean": 45.0,
    "small_heel_toe_y": -0.08,
}
```

値は次の方法で求めます。

| 保存名 | 1動作から集約する値 | 基準 |
|---|---|---|
| `shallow_squat_knee_angle` | 膝角度の最小値 | 学習用Good系列の90パーセンタイル |
| `large_torso_lean` | 体幹前傾角の最大値 | 学習用Good系列の90パーセンタイル |
| `small_heel_toe_y` | かかと―足先高さ差の最小値 | 学習用Good系列の10パーセンタイル |

テスト動画を基準値計算へ含めないでください。学習時とリアルタイム時で、関節、計算式、正規化、時系列の集約方法を一致させます。

## 6. コメント文体

腕立て伏せ版に合わせ、原因名だけでなく、利用者が次に行う動作を短い命令形で返します。

| 原因 | コメント |
|---|---|
| 上体前傾 | 上体を起こし、背中の傾きを小さくしてください |
| かかとの浮き | かかとを床につけたまま動作してください |
| 深さ不足 | もう少し深くしゃがんでください |
| 該当なし | 姿勢を改善してください |

同時に複数の問題がある場合は、腕立て伏せ版と同じく` / `でつなぎ、最大2件を返します。

## 7. UIを変更する場合

`sample_camera.py`は腕立て伏せ版と同じ画面構成です。左上に共通情報を表示し、Bad時だけ画面下部に日本語コメントを表示します。種目ごとの差は、生成するEngineとウィンドウ名だけです。

画面の色、配置、文字は`sample_camera.py`だけで変更できます。判定精度の変更でなければ、次のファイルを編集する必要はありません。

- `squat/analyzer.py`
- `squat/counter.py`
- `squat/engine.py`

Webアプリへ組み込む場合は、利用者またはセッションごとに別の`SquatEngine`を作成してください。複数人で1つのインスタンスを共有すると、フレーム履歴と回数が混ざります。

## 8. 実装確認項目

- `status`が`Good / Bad / Analyzing / No pose`のいずれか
- Bad時だけ骨格由来の改善コメントを生成する
- `bad_probability`が0～1
- 1往復で合計回数が1増える
- `total_count == good_count + bad_count`
- 動作中にGoodが一度でもあればGood回になる
- リセット時にMediaPipeのVIDEO時刻を戻さない
- 終了時に`engine.close()`を呼ぶ

## 9. macOSでの依存バージョン

- RandomForestの学習環境に合わせて`scikit-learn==1.6.1`を使用します。
- MediaPipeはPython 3.11のmacOS ARM64／Intelに対応する`0.10.21`を使用します。
- Pose Landmarker生成時にCPU delegateを明示し、macOSのMetal初期化を避けます。
- システムPythonへ直接インストールせず、プロジェクト専用の仮想環境を使用してください。

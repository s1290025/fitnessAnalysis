# 筋トレの姿勢分析チーム

### やったこと
* 筋トレデータから機械学習をし、姿勢の良し悪しを判定する。
* 良い姿勢の関節データと比較し、筋トレ姿勢のアドバイスコメントを出力する。
* 関節の上下をもとに、筋トレ回数をカウントする。

### コードは3種類
1. suquat_model_training.ipynb
    * スクワットの静止画データから姿勢の良し悪しを学習。また、アドバイスコメント用の基準値を作成。
    * Google Colabで動かせる。
2. pushup_model_training.ipynb
    * 腕立て伏せの動画データから姿勢の良し悪しを学習。また、アドバイスコメント用の基準値を作成。
    * Google Colabで動かせる。
3. fitness_posture_analysisフォルダ
    * UI部分。
    * フォルダ内のREADME.mdを見ながら、ローカルで動かす。（Macでは動きます。Windows等では動作確認していません。）
        * フォルダ内のsquat_analysisフォルダ、pushup_analysisフォルダのコードで、それぞれの筋トレの姿勢分析、コメント出力、回数カウントを行っている。
        * 詳細は、squat_analysisフォルダ、pushup_analysisフォルダそれぞれのREADME.mdを参照。

## 全体の流れ
1. 骨格データを入力する。
    * MediaPipeを使用し、画像や動画から骨格を抽出。
    * スクワットは、しゃがんだ静止画のデータをRandom Forestへ入力する。（suquat_model_training.ipynbにて学習済）
    * 腕立て伏せは、15フレーム分のデータをLSTMへ入力する。（pushup_model_training.ipynbにて学習済）
2. Random ForestかLSTMで、姿勢のGoodまたはBadを判定する。
3. Badの場合、角度や動作量を計算する。
4. 保存済みの基準値と比較する。（基準値の決め方については、pushup_model_training.ipynb内の説明を参照。）
5. 基準から外れた項目に対応するコメントを表示する。
6. 膝関節や腕関節の上下から、筋トレ回数をカウント。
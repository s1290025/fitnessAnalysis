# Fitness Posture Analysis

Webカメラを使用して、Push-up（腕立て伏せ）とSquat（スクワット）の姿勢をリアルタイムで分析するアプリです。

分析結果として、Good / Bad判定、改善コメント、Total / Good / Badの回数を表示します。

## 1. 必要な環境

- Python 3.11
- Webカメラ
- ブラウザ
- macOS または Windows

Python 3.11がインストールされていることを確認してください。

### macOS

```bash
python3.11 --version
```

### Windows

```powershell
py -3.11 --version
```

`Python 3.11.x` と表示されればOKです。

---

## 2. フォルダ構成

```text
fitness_posture_analysis/
├── app.py
├── requirements.txt
├── pushup_analysis/
└── squat_analysis/
```

ターミナルで `fitness_posture_analysis` ディレクトリに移動してから、以下の操作を行ってください。

---

## 3. 仮想環境を作成する

### macOS

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

ターミナルの先頭に `(.venv)` と表示されればOKです。

---

## 4. 必要なライブラリをインストールする

### macOS

```bash
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

### Windows

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` には、姿勢分析およびWeb UIに必要なライブラリ（TensorFlow、MediaPipe、OpenCV、Gradioなど）が含まれています。

---

## 5. アプリを起動する

### macOS

```bash
./.venv/bin/python app.py
```

### Windows

```powershell
.\.venv\Scripts\python.exe app.py
```

正常に起動すると、ターミナルに次のような表示が出ます。

```
Running on local URL: http://127.0.0.1:7860
```

---

## 6. ブラウザで開く

ブラウザで以下を開いてください。

```
http://127.0.0.1:7860
```

Chrome、Safari、Edgeなどからアクセスできます。

ブラウザからカメラの使用許可を求められた場合は、カメラへのアクセスを許可してください。

---

## 7. 使い方

1. `Exercise` から `Push-up` または `Squat` を選択します。
2. Cameraを開始します。
3. 身体全体が映るように、カメラに対して横向きになります。
4. 運動を開始します。
5. `Pose Analysis` に骨格と分析結果が表示されます。
6. `Status` に `Good` / `Bad` などの判定が表示されます。
7. `Feedback` に姿勢改善のコメントが表示されます。
8. `Total / Good / Bad` にカウント結果が表示されます。
9. 最初からやり直す場合は `Reset` を押してください。

---

## 8. アプリを終了する

アプリを実行しているターミナルで、

```
Ctrl + C
```

を押してください。

仮想環境から抜ける場合は、

```
deactivate
```

を実行してください。

---

## 補足

Squat起動時に、

```
squat_form_thresholds.joblib がないため、改善コメントには暫定値を使用します。
```

と表示される場合があります。

これは改善コメント用の学習済み基準値がない場合に暫定値を使用するという警告です。アプリ自体は起動できます。

# modelsフォルダ

次のファイルをこのフォルダへ配置してください。

- `squat_pose_classifier.joblib`：必須
- `pose_landmarker.task`：必須
- `squat_form_thresholds.joblib`：推奨

`squat_form_thresholds.joblib`は、テスト動画を含めず、学習用Good動画だけから算出した次の辞書を`joblib.dump()`で保存します。

```python
{
    "shallow_squat_knee_angle": 125.0,
    "large_torso_lean": 45.0,
    "small_heel_toe_y": -0.08,
}
```

数値は形式例です。実際にはGoodデータの10／90パーセンタイルから決定してください。ファイルがない場合は、プログラム内の暫定値で動作します。

# Mosquito Trajectory Prediction — DAYCON

모기의 과거 궤적(-400ms ~ 0ms, 11 포인트)을 기반으로 80ms 이후 3D 좌표를 예측하는 모델입니다.

## Approach

- 등속 외삽(Constant Velocity) 예측값을 기준으로 잔차(residual)를 학습
- 입력 피처: 상대 좌표 + 속도 + 가속도 (9 features)
- 모델: LSTM (2 layers, hidden 128)
- 평가 지표: Hit Rate @ r=0.01m

## Usage

```bash
pip install -r requirements.txt
python train.py
python inference.py  # outputs/submission.csv 생성
```

## Result

| Method | Hit Rate |
|--------|----------|
| Constant Velocity Baseline | 0.5788 |
| LSTM + Residual Learning | 0.5990 |

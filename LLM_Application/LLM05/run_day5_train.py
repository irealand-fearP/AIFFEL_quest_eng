# -*- coding: utf-8 -*-
"""
Day 5 섹션 2 — 캘리포니아 주택 가격 모델 학습 및 저장
"""
import os, sys, json
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

# ===== 2.1 데이터 로드 =====
print("=" * 60)
print("[2.1] 데이터 로드 및 탐색")
print("=" * 60)
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)
print(f"피처 크기: {X.shape}, 타겟 범위: {y.min():.2f} ~ {y.max():.2f} ($100,000)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"학습 {X_train.shape[0]:,}개 / 테스트 {X_test.shape[0]:,}개")

# ===== 2.2 정규화 =====
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
X_train_norm = (X_train - train_mean) / train_std
X_test_norm = (X_test - train_mean) / train_std

X_train_t = torch.FloatTensor(X_train_norm)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test_norm)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# ===== 2.3 모델 정의 및 학습 =====
print("\n" + "=" * 60)
print("[2.3] 모델 정의 및 학습")
print("=" * 60)


class HousingModel(nn.Module):
    def __init__(self, input_dim=8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


model = HousingModel(input_dim=8)
print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 50

model.train()
for epoch in range(1, EPOCHS + 1):
    running = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        running += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}/{EPOCHS} — Loss: {running/len(train_loader):.4f}")

# 테스트 평가
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t)
    test_mse = criterion(test_preds, y_test_t).item()
    mae = torch.abs(test_preds - y_test_t).mean().item()
print(f"\n테스트 MSE: {test_mse:.4f}")
print(f"테스트 MAE: {mae:.4f} ($100,000 단위)  =  ${mae*100000:,.0f}")

# ===== 2.4 저장 =====
print("\n" + "=" * 60)
print("[2.4] 모델 및 전처리 파라미터 저장")
print("=" * 60)
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/housing_model.pth")
print(f"저장: models/housing_model.pth ({os.path.getsize('models/housing_model.pth')/1024:.1f} KB)")
with open("models/housing_preprocessing.json", "w", encoding="utf-8") as f:
    json.dump({"mean": train_mean.tolist(), "std": train_std.tolist(),
               "feature_names": feature_names}, f, indent=2)
print("저장: models/housing_preprocessing.json")

# ===== 2.5 추론 모듈 테스트 =====
print("\n" + "=" * 60)
print("[2.5] app/housing_model.py 추론 테스트")
print("=" * 60)
from app.housing_model import HousingPredictor
predictor = HousingPredictor("models/housing_model.pth", "models/housing_preprocessing.json")
sample = {name: float(X_test[0, i]) for i, name in enumerate(feature_names)}
result = predictor.predict(sample)
print(f"입력 샘플 예측: ${result['predicted_price_usd']:,}  "
      f"(실제: ${int(y_test[0]*100000):,})")
print("\n[완료] Day 5 섹션 2 수행 완료")

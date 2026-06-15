# -*- coding: utf-8 -*-
"""
Day 1 — 섹션 5 실습: 학습된 모델을 저장하고 다시 불러오기
모델 배포 개론 01 (모두의연구소)의 섹션 5 워크플로우를 그대로 실행한다.
프로젝트 루트(model-serving-course)에서 실행하는 것을 전제로 한다.
"""
import os
import sys

# 항상 이 스크립트가 있는 폴더(프로젝트 루트)에서 동작하도록 고정
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


# ===== 모델 정의 (섹션 4와 동일) =====
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    # ===== 5.2 Step 2 — 모델 학습 =====
    print("=" * 60)
    print("[5.2] Step 2 — MNIST 모델 학습")
    print("=" * 60)

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 3
    torch.manual_seed(42)  # 재현성을 위해 시드 고정

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"학습 데이터: {len(train_dataset):,}장")
    print(f"테스트 데이터: {len(test_dataset):,}장")

    model = SimpleClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {running_loss/(batch_idx+1):.4f} Acc: {100.*correct/total:.1f}%")
        print(f"Epoch {epoch}/{EPOCHS} 완료 — Loss: {running_loss/len(train_loader):.4f}, "
              f"Acc: {100.*correct/total:.1f}%\n")

    # ===== 5.3 Step 3 — 세 가지 방식으로 저장 =====
    print("=" * 60)
    print("[5.3] Step 3 — 세 가지 방식으로 모델 저장")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)
    model_cpu = model.cpu()
    model_cpu.eval()

    test_input = test_dataset[0][0].unsqueeze(0)
    test_label = test_dataset[0][1]

    with torch.no_grad():
        original_output = model_cpu(test_input)
        original_pred = original_output.argmax(dim=1).item()
        original_conf = torch.softmax(original_output, dim=1).max().item()
    print(f"테스트 입력 크기: {test_input.shape}")
    print(f"정답 레이블: {test_label}")
    print(f"원본 모델 예측: {original_pred} (확신도: {original_conf:.4f})  "
          f"{'OK' if original_pred == test_label else 'X'}")

    torch.save(model_cpu.state_dict(), "models/mnist_state_dict.pth")
    print(f"  state_dict  저장: {os.path.getsize('models/mnist_state_dict.pth')/1024:.1f} KB")

    traced_model = torch.jit.trace(model_cpu, test_input)
    traced_model.save("models/mnist_traced.pt")
    print(f"  TorchScript 저장: {os.path.getsize('models/mnist_traced.pt')/1024:.1f} KB")

    torch.onnx.export(
        model_cpu, test_input, "models/mnist_model.onnx",
        export_params=True, opset_version=17,
        input_names=["image"], output_names=["prediction"],
        dynamic_axes={"image": {0: "batch_size"}, "prediction": {0: "batch_size"}},
    )
    print(f"  ONNX        저장: {os.path.getsize('models/mnist_model.onnx')/1024:.1f} KB")

    print("\n📁 models/ 폴더 내용")
    for fname in sorted(os.listdir("models")):
        size_kb = os.path.getsize(os.path.join("models", fname)) / 1024
        print(f"  {fname:<28} {size_kb:>8.1f} KB")

    # ===== 5.4 Step 4 — 불러오기 및 추론 검증 =====
    print("\n" + "=" * 60)
    print("[5.4] Step 4 — 불러오기 및 추론 검증")
    print("=" * 60)

    loaded_sd = SimpleClassifier(num_classes=10)
    loaded_sd.load_state_dict(torch.load("models/mnist_state_dict.pth", weights_only=True))
    loaded_sd.eval()
    with torch.no_grad():
        sd_output = loaded_sd(test_input)
        sd_pred = sd_output.argmax(dim=1).item()
    print(f"[state_dict]  예측: {sd_pred}, 원본과 일치: {torch.allclose(original_output, sd_output)}")

    loaded_ts = torch.jit.load("models/mnist_traced.pt")
    with torch.no_grad():
        ts_output = loaded_ts(test_input)
        ts_pred = ts_output.argmax(dim=1).item()
    print(f"[TorchScript] 예측: {ts_pred}, 원본과 일치: {torch.allclose(original_output, ts_output)}")

    import onnxruntime as ort
    session = ort.InferenceSession("models/mnist_model.onnx")
    onnx_output = session.run(["prediction"], {"image": test_input.numpy()})
    onnx_pred = int(np.argmax(onnx_output[0], axis=1)[0])
    onnx_match = np.allclose(original_output.numpy(), onnx_output[0], atol=1e-5)
    print(f"[ONNX]        예측: {onnx_pred}, 원본과 일치(오차허용): {onnx_match}")

    print("\n📊 직렬화 검증 결과 요약")
    print(f"  정답 레이블:       {test_label}")
    print(f"  원본 모델 예측:    {original_pred}")
    print(f"  state_dict 예측:  {sd_pred}  {'OK' if sd_pred == original_pred else 'X'}")
    print(f"  TorchScript 예측: {ts_pred}  {'OK' if ts_pred == original_pred else 'X'}")
    print(f"  ONNX 예측:        {onnx_pred}  {'OK' if onnx_pred == original_pred else 'X'}")
    if all(p == original_pred for p in [sd_pred, ts_pred, onnx_pred]):
        print("  => 세 가지 방식 모두 원본과 동일한 결과 (직렬화 검증 성공)")

    # ===== 5.5 Step 5 — 배치 추론 테스트 =====
    print("\n" + "=" * 60)
    print("[5.5] Step 5 — 배치 추론 테스트 (8장)")
    print("=" * 60)
    batch_images = torch.stack([test_dataset[i][0] for i in range(8)])
    batch_labels = [test_dataset[i][1] for i in range(8)]
    with torch.no_grad():
        sd_batch = loaded_sd(batch_images).argmax(dim=1).tolist()
        ts_batch = loaded_ts(batch_images).argmax(dim=1).tolist()
    onnx_batch = np.argmax(session.run(["prediction"], {"image": batch_images.numpy()})[0], axis=1).tolist()
    print(f"배치 입력 크기: {tuple(batch_images.shape)}")
    print(f"{'이미지':<7}{'정답':<6}{'state_dict':<12}{'TorchScript':<13}{'ONNX':<7}")
    print("-" * 48)
    for i in range(8):
        ok = "OK" if sd_batch[i] == ts_batch[i] == onnx_batch[i] == batch_labels[i] else "X"
        print(f"  #{i:<4}{batch_labels[i]:<6}{sd_batch[i]:<12}{ts_batch[i]:<13}{onnx_batch[i]:<7}{ok}")

    # ===== 5.6 Step 6 — 추론 함수 분리(app/model_utils.py) 테스트 =====
    print("\n" + "=" * 60)
    print("[5.6] Step 6 — app/model_utils.py 추론 함수 테스트")
    print("=" * 60)
    from app.model_utils import load_model, predict
    model_for_api = load_model("models/mnist_state_dict.pth")
    result = predict(model_for_api, test_input)
    print(f"  예측 클래스: {result['predicted_class']}")
    print(f"  확신도:     {result['confidence']}")
    print("  전체 확률:")
    for cls, prob in result["probabilities"].items():
        bar = "#" * int(prob * 40)
        print(f"    {cls}: {prob:.4f} {bar}")

    # ===== 5.7 프로젝트 최종 구조 =====
    print("\n" + "=" * 60)
    print("[5.7] 프로젝트 최종 구조")
    print("=" * 60)

    def show_tree(path, prefix="", max_depth=2, depth=0):
        if depth >= max_depth:
            return
        entries = sorted(os.listdir(path))
        entries = [e for e in entries if e not in
                   {".venv", ".venv_test", "__pycache__", ".ipynb_checkpoints"}
                   and not e.startswith(".venv_")]
        for i, e in enumerate(entries):
            full = os.path.join(path, e)
            connector = "└── " if i == len(entries) - 1 else "├── "
            if os.path.isdir(full):
                print(f"{prefix}{connector}{e}/")
                ext = "    " if i == len(entries) - 1 else "│   "
                show_tree(full, prefix + ext, max_depth, depth + 1)
            else:
                size = os.path.getsize(full)
                s = f"({size/1024:.1f} KB)" if size > 1024 else f"({size} B)"
                print(f"{prefix}{connector}{e} {s}")

    print("model-serving-course/")
    show_tree(".")
    print("\n[완료] Day 1 섹션 5 실습 수행 완료")


if __name__ == "__main__":
    main()

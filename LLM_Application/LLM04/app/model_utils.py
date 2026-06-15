"""
모델 로드 및 추론 유틸리티
FastAPI 엔드포인트가 이 모듈을 import하여 사용합니다.
"""

import torch
import torch.nn as nn
from torchvision import transforms


# ===== 모델 정의 =====
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


# ===== 전처리 파이프라인 =====
# 임의 이미지를 MNIST 입력 형식(1x28x28, 정규화)으로 변환합니다
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ===== 모델 로드 =====
def load_model(model_path: str, num_classes: int = 10) -> nn.Module:
    """저장된 state_dict를 불러와서 추론 가능한 모델을 반환합니다."""
    model = SimpleClassifier(num_classes=num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


# ===== 추론 =====
# 클래스 이름 매핑 (MNIST: 0~9)
CLASS_NAMES = [str(i) for i in range(10)]


def predict(model: nn.Module, input_tensor: torch.Tensor) -> dict:
    """전처리된 텐서를 입력받아 예측 결과를 딕셔너리로 반환합니다."""
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": round(float(probs[pred_idx].item()), 4),
        "probabilities": {
            CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(CLASS_NAMES))
        },
    }

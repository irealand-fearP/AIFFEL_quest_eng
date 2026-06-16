# -*- coding: utf-8 -*-
"""
Day 6 섹션 6 — 이미지 분류 API 통합 테스트 (FastAPI 서버가 떠 있어야 함)
6.3 인증없음(401) / 6.4 잘못된키(401) / 6.5 정상(200) / 6.6 잘못된형식(400) / 6.7 연속5장 / 6.8 Swagger
"""
import io
import requests
from torchvision import datasets

API = "http://127.0.0.1:8000"
KEY = "test-key-001"

print("=" * 60)
print("[섹션 6] 이미지 분류 API 통합 테스트")
print("=" * 60)

# 헬스체크
print(f"\n[헬스체크] {requests.get(f'{API}/health').json()}")

# MNIST 테스트 이미지 준비
test_dataset = datasets.MNIST(root="data", train=False, download=True)


def png_bytes(idx):
    img, label = test_dataset[idx]
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), label


# ── 6.3 인증 없이 요청 → 401 ──
r = requests.post(f"{API}/predict/image",
                  files={"file": ("test.png", b"fake", "image/png")})
print(f"\n[6.3] 인증 없음        → HTTP {r.status_code} (기대 401)")

# ── 6.4 잘못된 키 → 401 ──
r = requests.post(f"{API}/predict/image",
                  files={"file": ("test.png", b"fake", "image/png")},
                  headers={"X-API-Key": "wrong-key"})
print(f"[6.4] 잘못된 키        → HTTP {r.status_code} (기대 401)")

# ── 6.5 올바른 키 + MNIST 이미지 → 200 ──
data, label = png_bytes(0)
r = requests.post(f"{API}/predict/image",
                  files={"file": ("digit.png", data, "image/png")},
                  headers={"X-API-Key": KEY})
print(f"[6.5] 올바른 키+이미지 → HTTP {r.status_code} (기대 200)")
print(f"      정답 {label} / 응답 {r.json()}")

# ── 6.6 잘못된 파일 형식 → 400 ──
r = requests.post(f"{API}/predict/image",
                  files={"file": ("test.txt", b"this is not an image", "text/plain")},
                  headers={"X-API-Key": KEY})
print(f"[6.6] 잘못된 형식(txt) → HTTP {r.status_code} (기대 400)")

# ── 6.7 연속 5장 ──
print("\n[6.7] 연속 추론 테스트 (5장)")
for i in range(5):
    data, label = png_bytes(i)
    r = requests.post(f"{API}/predict/image",
                      files={"file": (f"digit_{i}.png", data, "image/png")},
                      headers={"X-API-Key": KEY})
    j = r.json()
    pred = j.get("label", "?")
    conf = j.get("confidence", 0)
    mark = "OK" if str(label) == str(pred) else "X"
    print(f"  이미지 {i}: 정답={label}, 예측={pred}, 확신도={conf:.4f} {mark}")

# ── 6.8 Swagger UI(/docs) 접근 확인 ──
d = requests.get(f"{API}/docs")
o = requests.get(f"{API}/openapi.json").json()
print(f"\n[6.8] Swagger UI(/docs) → HTTP {d.status_code} (기대 200)")
print(f"      문서 제목: {o.get('info', {}).get('title')}  /  엔드포인트: {list(o.get('paths', {}).keys())}")
print("      → 브라우저 http://localhost:8000/docs 에서 x-api-key=test-key-001 로 직접 테스트 가능")
print("\n[완료] Day 6 섹션 6 수행 완료")

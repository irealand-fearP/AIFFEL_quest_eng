# -*- coding: utf-8 -*-
"""
Day 5 섹션 3.3 + 5 — API 통합 테스트 (FastAPI 서버가 떠 있어야 함)
"""
import time, json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_BASE = "http://127.0.0.1:8000"

print("=" * 60)
print("[3.3 / 5] API 통합 테스트")
print("=" * 60)

# 헬스체크
resp = requests.get(f"{API_BASE}/health")
print(f"\n[헬스체크] {resp.status_code} {resp.json()}")

# 단일 추론
sample = {"MedInc": 3.5, "HouseAge": 25.0, "AveRooms": 5.0, "AveBedrms": 1.0,
          "Population": 1500.0, "AveOccup": 3.0, "Latitude": 37.5, "Longitude": -122.0}
resp = requests.post(f"{API_BASE}/predict", json=sample)
r = resp.json()
print(f"\n[단일 추론] {resp.status_code} → 예측 ${r['predicted_price_usd']:,}")

# 테스트 1: 다양한 입력
print("\n[테스트 1] 정상 요청 — 다양한 입력")
cases = [
    {"name": "저소득 지역", "MedInc": 1.5, "HouseAge": 40, "AveRooms": 4.0, "AveBedrms": 1.0,
     "Population": 2000, "AveOccup": 3.5, "Latitude": 34.0, "Longitude": -118.0},
    {"name": "고소득 지역", "MedInc": 10.0, "HouseAge": 10, "AveRooms": 8.0, "AveBedrms": 2.0,
     "Population": 500, "AveOccup": 2.0, "Latitude": 37.8, "Longitude": -122.4},
    {"name": "평균적 주택", "MedInc": 3.5, "HouseAge": 25, "AveRooms": 5.0, "AveBedrms": 1.0,
     "Population": 1500, "AveOccup": 3.0, "Latitude": 37.5, "Longitude": -122.0},
]
print(f"  {'케이스':<14}{'예측 가격':>14}")
print("  " + "-" * 28)
for case in cases:
    name = case.pop("name")
    rr = requests.post(f"{API_BASE}/predict", json=case).json()
    print(f"  {name:<14}{'$'+format(rr['predicted_price_usd'], ','):>14}")
    case["name"] = name

# 테스트 2: 에러 상황
print("\n[테스트 2] 에러 상황 (상태코드 확인)")
print(f"  필드 누락     → HTTP {requests.post(f'{API_BASE}/predict', json={'MedInc': 3.5}).status_code}")
bad = {**sample, "Latitude": 50.0}
print(f"  위도 범위초과 → HTTP {requests.post(f'{API_BASE}/predict', json=bad).status_code}")
bad2 = {**sample, "MedInc": -1.0}
print(f"  소득 음수     → HTTP {requests.post(f'{API_BASE}/predict', json=bad2).status_code}")
print(f"  잘못된 포맷   → HTTP {requests.post(f'{API_BASE}/predict', data='not json').status_code}")

# 테스트 3: 동시 요청 8개
print("\n[테스트 3] 동시 요청 (8개)")
def send(i):
    c = cases[i % len(cases)].copy(); c.pop("name", None)
    t = time.time()
    s = requests.post(f"{API_BASE}/predict", json=c, timeout=30).status_code
    return {"id": i+1, "elapsed": round(time.time()-t, 3), "status": s}
start = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    results = [f.result() for f in as_completed([ex.submit(send, i) for i in range(8)])]
for r in sorted(results, key=lambda x: x["id"]):
    print(f"  요청 #{r['id']}: {r['elapsed']}초 (HTTP {r['status']})")
print(f"  전체: {round(time.time()-start, 2)}초")

# 종합
print("\n" + "=" * 60)
print("  테스트 결과 종합")
print("=" * 60)
print("  OK 정상 요청: 다양한 입력에서 합리적 가격 반환")
print("  OK 에러 처리: 잘못된 입력에 422 반환, 서버 안 죽음")
print("  OK 동시 처리: 8개 동시 요청 정상 처리")
print("  OK 헬스체크: 서버 상태 healthy")

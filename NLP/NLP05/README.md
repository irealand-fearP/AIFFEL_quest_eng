# AIFFEL Campus Online Code Peer Review Templete
- 코더 : 코더의 이름을 작성하세요.
- 리뷰어 : 리뷰어의 이름을 작성하세요.


# PRT(Peer Review Template)
- [x ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
        - 중요! 해당 조건을 만족하는 부분을 캡쳐해 근거로 첨부
        - <img width="1057" height="308" alt="2026-05-22 14 53 28" src="https://github.com/user-attachments/assets/c7cdc86f-1318-46e5-8f0c-67d9db0c62e5" />

        - <img width="1003" height="224" alt="2026-05-22 14 46 04" src="https://github.com/user-attachments/assets/16487712-cdaa-4312-9769-d8f3308918f7" />
<img width="1057" height="183" alt="2026-05-22 14 46 13" src="https://github.com/user-attachments/assets/6fde8ad4-a0e0-4b7f-8713-342d75e53eac" />

    
- [ x]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭을 왜 핵심적이라고 생각하는지 확인
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드의 기능, 존재 이유, 작동 원리 등을 기술했는지 확인
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - <img width="722" height="349" alt="2026-05-22 14 47 31" src="https://github.com/user-attachments/assets/b8aa0b96-e31d-41d8-934a-e927abaeff45" />
이 프로젝트의 평가기준 3번(bucketing)을 직접 구현한 부분이고,
  transformers 5.9.0에서 `group_by_length=True` 인자가 제거되어 우회가 필요했던
  가장 까다로운 지점입니다.
- **doc string 확인**: 클래스 doc string에 동작 순서(정렬 → mega-batch → 셔플 → 분할)가
  기술되어 있어, `group_by_length` 한 줄이 내부에서 무슨 일을 하는지 코드 레벨로 이해됩니다.
- **이해도**: 동적 패딩(`DataCollatorWithPadding`)이 사전 조건이고, 그 위에 길이 그룹핑이
  얹히는 구조라는 점이 주석으로 연결되어 잘 이해되었습니다.
        
- [x ]  **3. 에러가 난 부분을 디버깅하여 문제를 해결한 기록을 남겼거나
새로운 시도 또는 추가 실험을 수행해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 프로젝트 평가 기준에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - keyword 'group_by_length'
        - Trainer가 알아서 "비슷한 길이끼리 묶어서 배치 만들기"를 해줬거든. 내부적으로 어떤 sampler를 쓸지 자동으로 결정
        - 환경이 transformers 5.9.0인데, 이 버전에서 group_by_length 인자가 TrainingArguments에서 제거됐어. 그래서 직접 우회주입
        - <img width="1105" height="359" alt="2026-05-22 14 58 18" src="https://github.com/user-attachments/assets/53d54da9-cb4a-45a1-8240-bade2851f866" />

        
- [x ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        
- [ x]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화/모듈화했는지 확인
        - 중요! 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부
        - 토큰화 시 패딩 ->> padding 생략 + DataCollatorWithPadding  정적 패딩 대비 메모리·속도 ↑
        - <img width="1123" height="316" alt="2026-05-22 15 05 54" src="https://github.com/user-attachments/assets/fa936970-a3ba-4181-b834-4cfeaa0eb1fb" />
        <img width="455" height="242" alt="2026-05-22 15 08 02" src="https://github.com/user-attachments/assets/ea8c2b00-759e-48e1-8c09-45b58a06a95f" />

        정적 = 고정값(128)으로 무조건 채움 → 짧은 문장은 PAD 낭비.
        동적 = 배치마다 그 안의 최대 길이에만 맞춤 → 낭비 최소.
        우리는 동적 패딩을 썼고, 그게 bucketing이 효과를 내기 위한 전제 조건이었다.




# 회고(참고 링크 및 코드 개선)

이 프로젝트의 가장 큰 강점은 "데이터 분석(STEP 1)이 끝까지 모든 결정을 끌고 갔다"는
일관된 스토리라인이다. 길이 분포 95%값(107자)에서 max_length=128을 도출하고,
max/median 5.4배라는 편차가 STEP 5 bucketing 효과 예측까지 쭈욱 이어갔음.



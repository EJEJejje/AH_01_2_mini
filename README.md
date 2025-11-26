📌 1. 프로젝트 개요

> 본 프로젝트는 **건강검진 데이터(7000명)**를 활용해
이상 여부(0/1)를 분류하는 이진 분류 모델을 개발하는 작업이다.

목표는 다음과 같다:
	•	Validation Accuracy 최대화
	•	일반화 성능 확보
	•	과적합 최소화
	•	Test 데이터에 대한 안정적 예측 수행

평가지표는 Accuracy다.

⸻

📊 2. 데이터 분석 (EDA 요약)

✔ 데이터 구조
	•	Train: 7,000개
	•	Test: 3,000개
	•	Feature: 총 16개 모두 수치형
	•	결측치 없음
	•	ID, label 제외하고 학습

✔ 주요 Feature
	•	나이, 키, 몸무게, BMI
	•	혈압, 공복 혈당
	•	콜레스테롤 계열 (총콜, HDL, LDL, 중성 지방)
	•	간 효소율, 혈청 크레아티닌
	•	헤모글로빈
	•	시력, 충치, 요 단백

✔ Feature 중요도 (RandomForest 기준 Top 변수)

순위
변수명
1
헤모글로빈
2
키(cm)
3
중성 지방
4
간 효소율
5
LDL(저밀도지단백)
➡ 혈액·간·신장 관련 지표가 가장 큰 영향을 미침

🤖 3. 모델링 전략

모델링은 아래 순서로 진행되었다.
	1.	Baseline 모델 학습 (RF, XGB)
	2.	RandomForest 하이퍼파라미터 세부 튜닝
	3.	XGBoost 하이퍼파라미터 실험
	4.	Soft Voting 앙상블
	5.	Threshold 튜닝 (0.40~0.60 grid search)
	6.	OOF 기반 과적합 점검
	7.	최종 모델 선정

⸻

🧪 4. 단일 모델 성능 정리

✔ RandomForest
항목
값
n_estimators
1100
max_depth
None
min_samples_split
4
min_samples_leaf
1
	•	Train ACC: ~0.99
	•	Valid ACC: 0.7471

📌 과적합이 매우 강하게 발생
(하지만 Valid 성능은 모든 모델 중 가장 높았음)
✔ XGBoost

항목
값
n_estimators
700
max_depth
6
learning_rate
0.07
subsample
0.9
	Train ACC: ~0.88
	•	Valid ACC: 0.7350

📌 단일 성능은 RF보다 낮지만
📌 앙상블에서 큰 개선 효과가 있었음.

⚡ 5. Soft Voting 앙상블 실험

✔ 탐색 범위
	•	Weight: RF 0.60~0.90, XGB 나머지
	•	Threshold: 0.40~0.60

✔ 결과

다수 실험 중 가장 높은 Valid Accuracy는 다음 조합에서 나왔다:

항목
값
RF Weight
0.75
XGB Weight
0.25
Threshold
0.47
⭐ Best Valid Accuracy
0.7586
🔍 6. 과적합 분석 및 개선 작업

✔ 과적합 확인
	•	RF Train ACC: 0.99
	•	RF Valid ACC: 0.74
	•	XGB Train ACC: 0.88
	•	XGB Valid ACC: 0.73

➡ 매우 강한 과적합 패턴.

✔ 과적합 완화 시도
	1.	RF
	•	max_depth=10 제한
	•	leaf 최소 샘플 증가
→ 일반화는 증가했으나 Valid 점수 감소
	2.	XGB
	•	depth 감소
	•	subsample 0.85
	•	reg_lambda 강화
→ 과적합 감소했으나 점수는 하락
	3.	CatBoost 추가 실험
→ 최고 성능 갱신 실패
	4.	OOF 기반 Ensemble
→ 일반화는 더 좋아졌지만 최고 Valid 기준에서는 0.74 수준

📌 결론

해당 데이터는 과적합을 일정 부분 허용할 때 점수가 높아지는 구조
규제를 강하게 넣을수록 점수가 떨어지는 특성이 있음

⸻

🏆 7. 최종 선택된 모델

✔ 최종 모델: RF + XGB Soft Voting
	•	RF 비중: 0.75
	•	XGB 비중: 0.25
	•	최적 Threshold: 0.47

✔ 최종 Validation Accuracy

0.7586

✔ Test 데이터 예측 방식
	1.	전체 7000개로 최종 모델 재학습
	2.	확률값을 w_rf=0.75 + w_xgb=0.25 로 결합
	3.	0.47 기준으로 이진 분류
	4.	제출 파일 생성

⸻

🧾 8. 요약 결론 
	•	RF가 가장 강력한 단일 모델
	•	XGB는 보조 역할로 앙상블 시 시너지
	•	Soft Voting + Threshold 튜닝이 점수를 가장 많이 올림
	•	과적합을 완전히 제거하면 오히려 성능 하락
	•	최종 Valid 최고 점수: 0.7586
	•	전반적인 실험 중 가장 안정적 성능

⸻

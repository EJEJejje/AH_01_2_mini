📌 1. 프로젝트 개요

본 프로젝트는 **건강검진 데이터(7,000명)**를 활용하여
개인의 이상 여부(0/1) 를 분류하는 머신러닝 모델을 개발하는 작업이다.

🎯 목표
	•	Validation Accuracy 최대화
	•	일반화 성능 확보
	•	과적합 최소화
	•	Test 데이터에 대한 안정적 예측

📈 평가지표: Accuracy

📊 2. 데이터 분석 (EDA 요약)

✔ 데이터 구조
	•	Train: 7,000개
	•	Test: 3,000개
	•	Feature: 16개 (모두 수치형)
	•	결측치 없음
	•	ID, label 제외하고 학습

✔ 주요 Feature 구성
	•	신체 지표: 나이, 키, 몸무게, BMI
	•	혈액 지표: 혈압, 공복 혈당
	•	지질: 총콜레스테롤, HDL, LDL, 중성 지방
	•	간 기능: 간 효소율
	•	신장/기타: 혈청 크레아티닌, 헤모글로빈, 시력, 요 단백, 충치

✔ Feature 중요도 (RandomForest 기준 Top 5)

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

➡ 전반적으로 혈액·간·지질 관련 지표들이 label에 가장 큰 영향

⸻

🤖 3. 모델링 전략

전체적인 실험 흐름은 다음과 같다:
	1.	Baseline 모델 실험 (RF, XGB)
	2.	RandomForest 하이퍼파라미터 튜닝
	3.	XGBoost 하이퍼파라미터 실험
	4.	Soft Voting 앙상블
	5.	Threshold 최적화 (0.40~0.60 탐색)
	6.	OOF 기반 교차검증 실험
	7.	최종 모델 선정 및 Test 예측

⸻

🧪 4. 단일 모델 실험 결과

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

⚠ 강한 과적합이 있지만 단일 성능은 두 모델 중 가장 높음.

⸻

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

	•	Train ACC: ~0.88
	•	Valid ACC: 0.7350

📌 단일 모델은 RF보다 낮지만
📌 앙상블에서 큰 시너지 발휘

⸻

⚡ 5. Soft Voting 앙상블 실험

✔ 탐색 범위
	•	Weight (RF): 0.60 ~ 0.90
	•	Weight (XGB): 1 - RF
	•	Threshold: 0.40 ~ 0.60

✔ 최종 결과

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
➡ 여러 세팅 중 가장 높은 Validation 성능 달성

🔍 6. 과적합 분석 & 개선 과정

✔ 과적합 확인
모델
Train ACC
Valid ACC
RF
0.99
0.74
XGB
0.88
0.73
➡ 명확한 과적합 패턴
✔ 과적합 완화 실험
	1.	RF 규제 강화
	•	max_depth=10, min_samples_leaf 증가
→ 일반화 향상했지만 점수 하락
	2.	XGB 규제 강화
	•	depth 감소, subsample 조정
→ 과적합 완화는 성공, 성능 하락
	3.	CatBoost 추가 실험
→ 최고 점수 갱신 실패
	4.	OOF 기반 Ensemble
→ 안정적이었으나 최고 점수(0.7586)에 도달 못함

⸻

📌 결론

이 데이터에서는 과적합을 많이 줄이면 오히려 성능이 떨어지는 구조
👉 적당한 과적합을 허용했을 때 성능이 가장 높아짐
👉 Soft Voting + Threshold 튜닝이 점수를 올린 핵심

⸻

🏆 7. 최종 선택된 모델

✔ 최종 앙상블 구조

RandomForest + XGBoost Soft Voting
	•	RF Weight = 0.75
	•	XGB Weight = 0.25
	•	Threshold = 0.47
	•	Final Valid Accuracy = 0.7586

✔ Test 예측 방식
	1.	전체 Train(7000개)로 RF & XGB 재학습
	2.	확률값을 weighted sum
	3.	Threshold=0.47 로 이진 분류
	4.	submission.csv 생성

⸻

🧾 8. 최종 요약
	•	RF가 가장 강력한 단일 모델
	•	XGB는 RF를 보완해 앙상블 성능 극대화
	•	Soft Voting + Threshold 탐색이 성능 상승의 주요 요인
	•	과적합 규제는 성능 저하로 이어짐
	•	최종 Valid Accuracy 0.7586
	•	지금까지 실험한 모델 중 가장 안정적이고 높은 성능
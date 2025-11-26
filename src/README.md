## 📌 프로젝트 설명: RandomForest 기반 분류 모델
1. 프로젝트 개요

이 프로젝트는 주어진 학습 데이터(train.csv)를 기반으로
RandomForestClassifier 모델을 학습하여 label을 예측하는 분류 모델을 만드는 작업입니다.
최종적으로 테스트 데이터(test.csv)에 대한 예측값을 생성하여 submission.csv 파일로 저장합니다.

2. 사용 라이브러리
import pandas as pd
import numpy as np
import seaborn as sns
import random
import os
import warnings

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

3. Seed 고정 (재현성 확보)
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

4. 데이터 로드 및 전처리
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.drop(['ID', 'label'], axis=1)
y_train = train['label']
x_test = test.drop(['ID'], axis=1)



## ID 컬럼 제거

label은 정답값이므로 분리하여 y_train에 저장

5. 랜덤 포레스트 모델 생성 및 학습
✔ 하이퍼파라미터 설정

n_estimators=290

max_depth=6

min_samples_split=5

min_samples_leaf=5

random_state=42

✔ 모델 코드
model = RandomForestClassifier(
    n_estimators=290,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=5,
    random_state=42
)

model.fit(x_train, y_train)

6. 테스트 데이터 예측 & 제출 파일 생성
pred = model.predict(x_test)

submit = pd.read_csv('sample_submission.csv')
submit['label'] = pred
submit.to_csv('submission.csv', index=False)

7. 검증 데이터 정확도 평가 (Train/Test Split)
X_train, X_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)

print("검증 데이터 정확도:", accuracy)

## 8. 결과 요약

랜덤포레스트 모델을 사용해 안정적인 예측 성능 확보

검증 정확도 출력

제출 파일 submission.csv 성공적으로 생성

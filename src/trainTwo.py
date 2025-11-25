import pandas as pd
import numpy as np
import seaborn as sns
import random
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정


# pd.read_csv() 함수를 사용해서 데이터를 읽어오는 코드입니다.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 데이터를 확인하기 위해 head() 함수를 사용합니다.
train.info()
x_train = train.drop(['ID','label'], axis=1)
y_train = train['label']
x_test = test.drop(['ID'], axis=1)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=5,
    random_state=42
)

model.fit(x_train, y_train)
pred = model.predict(x_test)

submit = pd.read_csv('sample_submission.csv')
submit['label'] = pred
submit.to_csv('submission.csv', index=False)


# 데이터 분할 (훈련:검증 = 8:2)
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
# 검증 데이터 예측
val_pred = model.predict(X_val)

# 정확도 계산
accuracy = accuracy_score(y_val, val_pred)
print("검증 데이터 정확도:", accuracy)
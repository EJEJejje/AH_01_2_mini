import pandas as pd
import numpy as np
import random
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정


# pd.read_csv() 함수를 사용해서 데이터를 읽어오는 코드입니다.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 데이터를 확인하기 위해 head() 함수를 사용합니다.
print(train.head(5))

x_train = train.drop(['ID','label'], axis=1)
y_train = train['label']

x_test = test.drop(['ID'], axis=1)
# 모델 인자에 random_state를 넣음으로써 시드고정의 효과를 얻을 수 있습니다.
model = DecisionTreeClassifier(random_state = 42)
model.fit(x_train, y_train)
pred = model.predict(x_test)

submit = pd.read_csv('sample_submission.csv')
submit['label'] = pred
submit.head()
print(submit.head(5))

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv('submission.csv', index=False)




import pandas as pd
import numpy as np
import seaborn as sns
import random
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
else:  # Linux (Colab 등)
    rc('font', family='NanumGothic')
# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False



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
    n_estimators=400,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(x_train, y_train)
pred = model.predict(x_test)

submit = pd.read_csv('sample_submission.csv')
submit['label'] = pred
submit.head()
print(submit.head(5))

# 예측한 결과를 파일로 저장합니다. index 인자의 값을 False로 설정하지 않으면 제출이 정상적으로 진행되지 않습니다.
submit.to_csv('submission.csv', index=False)
# 기술통계량을 확인하기위해 describe() 함수를 사용합니다.  
train.describe()
# 데이터프레임이 train이라면 다음과 같이 사용
sns.countplot(x='label', data=train)


# 기술통계량
train.describe()

# countplot
sns.countplot(x='label', data=train)
plt.show()

# Heatmap
numeric_train = train.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_train.corr(), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()



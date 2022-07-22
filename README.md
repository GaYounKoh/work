# work
internship

# 220719
0. 인사!!!
1. mac에서 vscode하는 방법 찾아야 함. (그 전에 os에 대해 고민 해봐야함. mac 너무 어려움. mac은 뭐라도 익히고 난 다음에..., 아주 기본적인 단축키 조차 못써서 답답함.)
2. python으로 SQL하는 코드 익히기 (불러오는 코드 정도만 익히면 판다스는 문제 없음.(? 아마도...? 설마..?))
3. ip 회사 서버에 연결
4. 회사 바로 앞 아샷추의 위치를 파악해봐야함. (일단 이디야는 없음.)
5. 점심시간 1시간, 12:30 ~ 13:30
6. 대회에 대해 익히기
7. AImers는 빨리 전화해봐야할듯.
8. 금요일 빨리 집
9. team: 동하님 기민님

[pymysql 사용법](https://www.fun-coding.org/mysql_basic6.html)
[json file이 도대체 뭐죠???, vsc의 예시 사진 함께 있음.](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=demonic3540&logNo=221277604043)

1) json : Java Script Object Notation <br>
2) 단순히 데이터를 표시하는 방법 <br>
3) json파일이 가지고 있는 데이터를 받아서 객체나 변수에 할당해서 사용하기 위함. <br>
 <br>
4) [json의 구조] <br>
 <br>
    1. Object (객체) <br>
    - name/value의 순서쌍으로 set <br>
    - {}로 정의됨. ex) {'이름':'홍길동'} <br>
 <br>
    2. Array (배열) <br>
    - ex) [10, 'arr', 32] <br>
 <br>
5) json의 예 <br>
{'이름':'홍길동', <br>
'나이':22, <br>
'특기':["배구","야구"]} <br>
 <br>
6) json은 왜 쓰는가? <br>
다른 포맷에 비래 경량화된 데이터 포맷. <br>
 <br>
7) json parsing? <br>
A로부터 B가 `.json` 파일을 받았다고 하면 이 파일에 담긴 data를 찾아 객체나 변수에 할당하기 위해 `.json` 파일 내에서 특정 data만을 가져와야 함. <br>
`.json` 파일 내의 특정 data만 추출하는 것을 의미. <br>


- cursor의 fetchall()메서드는 모든 데이터를 한 번에 클라이언트로 가져올 때 사용됨.
- fetchone()은 한번 호출에 하나의 Row 만을 가져올 때 사용
    - fetchone()을 여러 번 호출하면, 호출 때 마다 한 Row 씩 데이타를 가져오게 된다
- fetchmany(n) 메서드는 n개 만큼의 데이타를 한꺼번에 가져올 때 사용
[fetchall() 등 pymysql에 대한 설명](http://pythonstudy.xyz/python/article/202-MySQL-%EC%BF%BC%EB%A6%AC)

```python
import pymysql

sql = utils_ua.load_table('sql')
# conn, execute(), cursor(), 아래의 포맷 자주 사용되니 구글링을 통해 익힐 것.

with open('json.json', 'r', encoding = 'UTF8') as f:
    CONFIG = json.load(f) # 이것이 json 파싱

## set param
db_param = {'user':'유저',
            'password':'비밀번호',
            'host':'호스트네임',
            'charset':'뭐였더라',
            'db':'데이터베이스'}


conn = pymysql.connect(**db_param) # SQL에 연결
curs = conn.cursor() # cursor 객체 (커서 객체, db커서가 fetch 동작 관리함.)

cuery = 'show tables' # table의 이름
curs.execute(cuery) # 쿼리를 SQL에 전달
show_tables = curs.fetchall()

#-------------------------------------------------------------

tmp = agreement[(agreement["cpt_id"] == 숫자숫자)] #cpt_id가 숫자숫자 데이터만
tmp = tmp.drop(["불필요", "불필요", "불필요", "불필요"], axis = 1) # 불필요한 열은 빼


# 특정 데이터 뽑는 함수 (주어진 데이터1의 맨 끝 두 개 데이터)
def get_info(data, select = 1):
    result1 = data.split(',')[-2] # 주어진 데이터를 컴마로 나누고 끝에서 2번째 데이터
    result2 = data.split(',')[-1]
    if select == 1:
        return result1
    else:
        return result2


# 위의 함수를 데이터의 특정 열에 적용.
tmp["team_format"] = tmp["ㅊㅊ"].apply(lambda x : get_info(x, 1)) # 개인 or 팀
tmp["team_name"] = tmp["ㅊㅊ"].apply(lambda x : get_info(x, 2)) # 팀이라면 팀명

tmp.drop(["ㅊㅊ"], axis = 1, inplace = True) # 사용한 데이터는 drop으로 떨구기.
```



# 220720
### error message
```
Expected 2D array, got 1D array instead:
array=[val val val val val val val].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```
는 보통 X shape이 틀렸을 때 나옴.

### cpu 개수 확인 방법
```python
import os
os.cpu_count()
```

[공식 문서 MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) <br>

[Multi Output Regression new metric](https://www.kaggle.com/code/samanemami/multioutput-regression-new-metric/notebook) <br>

- 파이프라인(pipeline)을 사용한 grid search <br>
[[파이썬] MultiOutputRegressor or Classifier의 모델 튜닝 / Random Grid Search](https://koreapy.tistory.com/941) <br>
[grid search over mmulti output regressor](https://stackoverflow.com/questions/43532811/gridsearch-over-multioutputregressor) <br>

[완전 아래 쪽에 multioutput 예시 코드](https://machinelearningmastery.com/multi-output-regression-models-with-python/) <br>

[완전 아래 쪽에 multi-output regression, 다중 출력 회귀](https://conanmoon.medium.com/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EA%B3%BC%ED%95%99-%EC%9C%A0%EB%A7%9D%EC%A3%BC%EC%9D%98-%EB%A7%A4%EC%9D%BC-%EA%B8%80%EC%93%B0%EA%B8%B0-%EC%BA%A1%EC%8A%A4%ED%86%A4-%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-1-4%EC%A3%BC%EC%B0%A8-5690591dba43) <br>

[torch라서 일단 보다 말았음. Pytorch) multioutput Regression 구현해보기](https://data-newbie.tistory.com/845) <br>


[train - val - test](https://modern-manual.tistory.com/19) <br>

[gpu 관련](https://driz2le.tistory.com/270) <br>
[gpu 지정 사용](https://jimmy-ai.tistory.com/121) <br>
```python
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()
# 출처: https://koreapy.tistory.com/239 [py:티스토리] # 별 도움 안됨

import tensorflow as tf
# tf.config.list_physical_devices('GPU')
tf.config.experimental.list_physical_devices('GPU')

import torch
torch.cuda.device_count()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.current_device()
torch.cuda.device(0)





# gpu 지정 사용 코드
# GPU 사용을 원하는 경우
with tf.device('/device:GPU:0'): 
    # 원하는 코드 작성(들여쓰기 필수)

# CPU 사용을 원하는 경우
with tf.device('/cpu:0'): 
    # 원하는 코드 작성(들여쓰기 필수)
```


[딥러닝, 층 쌓는 예시 코드](https://tensorflow.blog/%EC%BC%80%EB%9D%BC%EC%8A%A4-%EB%94%A5%EB%9F%AC%EB%8B%9D/3-6-%EC%A3%BC%ED%83%9D-%EA%B0%80%EA%B2%A9-%EC%98%88%EC%B8%A1-%ED%9A%8C%EA%B7%80-%EB%AC%B8%EC%A0%9C/) <br>

```python
from keras import models
from keras import layers

def build_model(): 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
```


- pipeline (파이프라인)
[파이프라인이란?, 전처리 파이프라인 구축해보기 1](https://gogetem.tistory.com/469) <br>
[파이프라인이란?, 전처리 파이프라인 구축해보기 2](https://rk1993.tistory.com/entry/Python-sklearnpipeline-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8Pipeline%EC%9D%B4%EB%9E%80) <br>

[머신러닝 파이프라인, 머신러닝 전후 코드 차이 보여줌. 굿굿](https://study2give.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8Pipeline) <br>
데이터 전처리와 모델 학습, 예측까지 한번에

[파이프라인 생성하는 일련의 과정 보여줌. 따라해보기](https://guru.tistory.com/50) <br>


### 모델 평가지표
```python
from sklearn.metrics import mean_squared_error as mse # 모델 평가 지표 scoring (mse)
from sklearn.metrics import r2_score as r2
mse(정답, 예측)
```


```python
from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor().get_params(deep = True)

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgbm
MultiOutputRegressor(lgbm.LGBMRegressor()).get_params()
```


# multi output regressor grid_search 시
- verbose=1로 줬을 경우 제일 마지막에 나와있는 time이 총 소요시간
- 튜닝하려고 골라놓은 각 하이퍼파라미터의 그 총 개수가 돌아가는 task 개수(??) 결정하는 게 맞음. (오늘 확실히 계산해봄.)
    - 예를 들어)
    ```python
    'estimator__n_estimators': np.linspace(50, 200, 16, dtype=int),
    'estimator__max_depth': np.linspace(30, 200, 18, dtype=int),
    'estimator__learning_rate': np.linspace(0.001, 0.1, 10, dtype = float),
    'estimator__min_child_weight': np.linspace(0.001, 0.1, 10, dtype=float)
    ```
    라면
    각 하이퍼 파라미터 마다 돌려보고싶은 수치의 개수? (즉 위에서부터 순서대로 16, 18, 10, 10를 의미)를 모두 더한 값이 최종 돌아가는 task의 수임.

- 한 번에 너무 많은 수의 task를 하려고 하지 말것.
    - 적당히 끊어서 하는게 더 효율적임.



[eval](https://bio-info.tistory.com/84)


### error message
``` python
The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```
|나 & 쓰라는 에런데 써도 안달라짐.

[파이썬 실습](https://hungryap.tistory.com/69) <br>


```python3
from sklearn.multioutput import MultiOutputRegressor
# from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV

# seed 고정
user_seed = 42
random.seed(user_seed) # seed 고정

param = {
    'estimator__n_estimators': np.linspace(60, 100, 5, dtype=int),
     'estimator__max_depth': np.linspace(5, 30, 6, dtype=int)
}

gs = GridSearchCV(estimator=MultiOutputRegressor(lgbm.LGBMRegressor()), 
                  param_grid=param, 
                  cv=2,
                  scoring = 'neg_mean_squared_error',
                  n_jobs = 16,
                 verbose = 1)
regr = gs.fit(X_train, Y_train)



print(regr.best_score_, regr.best_params_)
regr_model = regr.best_estimator_


regr_model.get_params()['estimator']


from sklearn.metrics import mean_squared_error as mse # 모델 평가 지표 scoring (mse)
from sklearn.metrics import r2_score as r2
Y_val_pred = regr_model.predict(X_val)
mse(Y_val, Y_val_pred)


Y_pred = regr_model.predict(X_test)
# 끝

```

# 220721
multioutput regressor로 한 번에 돌리기 보다는
y 각각에 대해 따로따로 돌려서 예측 결과 따로 따로 받는 것이 더 좋을 거라는 조언을 얻음.



# 220722
jupyter lab에서 LGBMRegressor 돌릴 때 파쳐 네임에 한글 있으면 안됨.
XGBRegressor 돌릴 때는 X_test까지도 feature네임에 한글 있으면 안됨.

```python
from sklearn.ensemble import RandomForestRegressor # decision tree 앙상블 모델, 배깅
Random_Forest = RandomForestRegressor()

import xgboost as xgb
XGB = xgb.XGBRegressor()

import lightgbm as lgbm
LGBM = lgbm.LGBMRegressor()
```

- 컬럼 이름 각각 바꾸기
```python
X_train.rename(columns = {before:after}, inplace = True)
```




- feature engineering 은 그냥 feature 정보를 가지고 feature를 추가 생성 하는 것을 의미하는 거였음. (그래서 전처리 단계에 포함됨.) <br>
- multioutput (multi label)의 경우 y끼리의 그림도 봐야함. (y끼리 플랏그리기) <br>
    -> 평가지표(metric) rmse, mse의 경우 target값이 큰 경우도 놓치면 안됨. 그걸 얼마나 잘 예측하는지로 점수가 갈리기 때문. 따라서 mse의 경우 이상치 함부로 제거하면 안됨. <br>
    -> 참고로 평가지표 mae의 경우는 값의 크기가 어떻든 모두 균등한 비율로 중요함. <br>
    <br>
- y끼리의 상관관계를 봐야함. <br>
- 예측한 y를 가지고 다음 y를 예측하는 방법을 배워놓고 잊고있었음. <br>
    -> 이게 y끼리의 상관관계를 보는 이유임. <br>

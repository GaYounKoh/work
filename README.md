# work
internship
<br>

## 221119 잊기전에하는_팀원_업데이트
[DS 대회파트 최종]
사수님 : 김동하 (1대 팀장님), 김기민 (2대 팀장님), 윤태양
인턴 : 고가연, 이승윤


## linux unzip code - 220816
- python 아니고 linux인 점 주의 <br>
```bash
unzip '~~~.zip' -d 'unzip한 file 저장할 path'
```
<br>

## 프린트 - 220809
위워크 홈페이지 > print/download?/os=Windows > 프린트 프로그램 설치 <br>
키카드만 탭하면 프린트 가능함. <br>

### 회사 주피터 랩 n_jobs = 16정도로 진행할 것. 내부 cpu 개수에 따라 조정해놨던 것
-220726
<br>

# file name 생성 규칙
- [0]은 언제나 활용할 수 있는 코드
- 나머지는 작업 할당 받은 순서
- [numbering] <file을 통해 알 수 있는 것과 관련해서 지은 이름> + (앞이 인턴 폴더에 저장된 코드와 이름이 다르면 <원본 이름>) + <주최 기업>
<br>

## [try except 구문] - 220809
- 완전 자동화를 꿈꾸는 구문임
- 조건 없이 일단 실행하고 실행 시 오류가 나면 다음으로 넘어가는 문법. <br>
    for문에서 try에 있는 구문을 실행했을 때 오류가 나면 except 구문에 continue 작성을 통해 다음 iter로 넘어가는 등의 처리가 가능하다. <br>
    - 자세한 코드 예시는 새로 업로드하는 파일 ([6] API_Crawling 함수로호출_try_except 구문 + visit_kor.ipynb)을 참고하시오. <br>

```python
## 예시 1)
## ci는 현재 경로에 없는 파일, cids 파일은 현재 경로에 존재하는 파일이다.
import pandas as pd
try:
    data = pd.read_csv('ci.csv')
except:
    data = pd.read_csv('cids.csv')
```
<br>

```python
## 예시 2) 자세한 코드는 새로 업로드하는 파일 ([6] API_Crawling 함수로호출_try_except 구문 + visit_kor.ipynb)을 참고하시오.
for enu, CID in tqdm(enumerate(cids[17004:])):
    try:
        tmp_df = api(CID)
    except:
        print(cids.index(CID))
        continue
        # except문에 동일한 요청이 아닌 다음 loop가 돌아야함.
        # 그러려면 continue를 해줘야 하는데 그럼 try부터 도는데, 그게 맞나..?
        # 말하다보니까 그게 맞다는걸 깨달음.
    
    ## concat & df update
    df = pd.concat([df, tmp_df], ignore_index = True)
```
<br>

## [평가 metric 보는 코드] - 220726
```python
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
# 평가방식들 보는 코드
```
<br>

## [df col to nonnull] - 220817
df의 col 기준으로 nonnull 만들기 <br>
```python
dat = dat.drop(dat[dat.imgurl1.isnull()].index | dat[dat.imgurl2.isnull()].index)
dat.to_csv('dat_nonnull.csv', index = False)
```
<br>


## [pymysql] - 220719
[pymysql 사용법](https://www.fun-coding.org/mysql_basic6.html)
- cursor의 fetchall()메서드는 모든 데이터를 한 번에 클라이언트로 가져올 때 사용됨.
- fetchone()은 한번 호출에 하나의 Row 만을 가져올 때 사용
    - fetchone()을 여러 번 호출하면, 호출 때 마다 한 Row 씩 데이타를 가져오게 된다
- fetchmany(n) 메서드는 n개 만큼의 데이타를 한꺼번에 가져올 때 사용
[fetchall() 등 pymysql에 대한 설명](http://pythonstudy.xyz/python/article/202-MySQL-%EC%BF%BC%EB%A6%AC) <br>

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
<br>


# 220719
0. 인사!!!
1. ~~mac에서 vscode하는 방법 찾아야 함. (그 전에 os에 대해 고민 해봐야함. mac 너무 어려움. mac은 뭐라도 익히고 난 다음에..., 아주 기본적인 단축키 조차 못써서 답답함.)~~ 포기하고 일단 내 노트북 쓰는중.
2. python으로 SQL하는 코드 익히기 (불러오는 코드 정도만 익히면 판다스는 문제 없음.(? 아마도...? 설마..?))
3. ~~ip 회사 서버에 연결~~ 연결 완료!, 집에서만 쓸 수 있음
4. ~~회사 바로 앞 아샷추의 위치를 파악해봐야함. (일단 이디야는 없음.)~~ 아이스티 사다놓음.
5. 점심시간 1시간, 12:30 ~ 13:30 ☆☆☆, 샌드위치집 위치 알아두기
6. 대회에 대해 익히기
7. AImers는 빨리 전화해봐야할듯.
8. ~~금요일 빨리 집~~ 일단 8시 전까지면 됨.
9. team: 동하님 기민님
<br>

## [json]
[json file이 도대체 뭐죠???, vsc의 예시 사진 함께 있음.](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=demonic3540&logNo=221277604043)

1) json : Java Script Object Notation <br>
2) 단순히 데이터를 표시하는 방법 <br>
3) json파일이 가지고 있는 데이터를 받아서 객체나 변수에 할당해서 사용하기 위함. <br>
 <br>
4) [json의 구조] <br>
 <br>
    1. Object (객체) <br>
    - name/value의 순서쌍으로 set <br>
    - { }로 정의됨. ex) {'이름':'홍길동'} <br>
 <br>
    2. Array (배열) <br>
    - ex) [10, 'arr', 32] <br>
 <br>
5) json의 예 <br>
```json
{'이름':'홍길동',
        '나이':22,
        '특기':["배구","야구"]}
```

6) json은 왜 쓰는가? <br>
다른 포맷에 비래 경량화된 데이터 포맷. <br>
 <br>
7) json parsing? <br>
A로부터 B가 `.json` 파일을 받았다고 하면 이 파일에 담긴 data를 찾아 객체나 변수에 할당하기 위해 `.json` 파일 내에서 특정 data만을 가져와야 함. <br>
`.json` 파일 내의 특정 data만 추출하는 것을 의미. <br>



# 220720
## [error message]
```
Expected 2D array, got 1D array instead:
array=[val val val val val val val].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```
는 보통 X shape이 틀렸을 때 나옴.

## [cpu 개수 확인 코드]
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


## [GPU 지정 코드]
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


## [딥러닝 층 쌓는 예시 코드]
[딥러닝, 층 쌓는 예시 코드](https://tensorflow.blog/%EC%BC%80%EB%9D%BC%EC%8A%A4-%EB%94%A5%EB%9F%AC%EB%8B%9D/3-6-%EC%A3%BC%ED%83%9D-%EA%B0%80%EA%B2%A9-%EC%98%88%EC%B8%A1-%ED%9A%8C%EA%B7%80-%EB%AC%B8%EC%A0%9C/) <br>

```python
from keras import models
from keras import layers

def build_model(): # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
```


- pipeline (파이프라인) <br>
[파이프라인이란?, 전처리 파이프라인 구축해보기 1](https://gogetem.tistory.com/469) <br>
[파이프라인이란?, 전처리 파이프라인 구축해보기 2](https://rk1993.tistory.com/entry/Python-sklearnpipeline-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8Pipeline%EC%9D%B4%EB%9E%80) <br>

    [머신러닝 파이프라인, 머신러닝 전후 코드 차이 보여줌. 굿굿](https://study2give.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8Pipeline) <br>
    데이터 전처리와 모델 학습, 예측까지 한번에

    [파이프라인 생성하는 일련의 과정 보여줌. 따라해보기](https://guru.tistory.com/50) <br>

<br>

## [모델 평가지표](https://scikit-learn.org/stable/modules/model_evaluation.html) - metrics
```python
from sklearn.metrics import mean_squared_error as mse # 모델 평가 지표 scoring (mse)
from sklearn.metrics import r2_score as r2
mse(정답, 예측)

# rmse
mse(정답, 예측, squared=False)

# multioutput 시 col 각각의 rmse 점수
mse(정답, 예측, multioutput='raw_values', squared=False)
```
<br>

```python
from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor().get_params(deep = True)

from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgbm
MultiOutputRegressor(lgbm.LGBMRegressor()).get_params()
```
<br>

```python
import random
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
# RandomForestRegressor().get_params(deep = True)

# seed 고정
user_seed = 42
random.seed(user_seed) # seed 고정

regr = MultiOutputRegressor(RandomForestRegressor(random_state=user_seed, n_jobs=16)).fit(X_train, Y_train) # verbose 너무 시끄러워서 끔.
Y_pred = regr.predict(X_test)

Y_pred

# 평가
from sklearn.metrics import mean_squared_error as mse # 모델 평가 지표 scoring (mse)
from sklearn.metrics import r2_score as r2
# mse(Y_test, Y_pred, multioutput='raw_values', squared=False) # rmse
mse(Y_test, Y_pred, squared=False) # rmse
r2(Y_test, Y_pred)
```
<br>

```python
import random
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
XGB = xgb.XGBRegressor(random_state=user_seed, n_jobs=16)
# XGB.get_params(deep = True)

# seed 고정
user_seed = 42
random.seed(user_seed) # seed 고정

regr = MultiOutputRegressor(XGB).fit(X_train, Y_train)
Y_pred = regr.predict(X_test)

Y_pred


# 평가
from sklearn.metrics import mean_squared_error as mse # 모델 평가 지표 scoring (mse)
from sklearn.metrics import r2_score as r2
# mse(Y_test, Y_pred, multioutput='raw_values', squared=False) # rmse
mse(Y_test, Y_pred, squared=False) # rmse : 
r2(Y_test, Y_pred) # : 
```
<br>

```python
from sklearn.multioutput import MultiOutputRegressor
import random
import lightgbm as lgbm
# MultiOutputRegressor(lgbm.LGBMRegressor()).get_params()

# seed 고정
user_seed = 42
random.seed(user_seed) # seed 고정

regr = MultiOutputRegressor(lgbm.LGBMRegressor(random_state=user_seed)).fit(X_train, Y_train)
Y_pred = regr.predict(X_test)

Y_pred


# 평가
from sklearn.metrics import mean_squared_error as mse # 모델 평가 지표 scoring (mse)
from sklearn.metrics import r2_score as r2
# mse(Y_test, Y_pred, multioutput='raw_values', squared=False) # rmse
mse(Y_test, Y_pred, squared=False) # rmse (계절, 월 추가) : 
r2(Y_test, Y_pred) # (계절, 월 추가) : 
```
<br>



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


### [error message]
``` python
The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
```
|나 & 쓰라는 에런데 써도 안달라짐.

[파이썬 실습](https://hungryap.tistory.com/69) <br>


```python
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
multioutput regressor로 한 번에 돌리기 보다는 <br>
y 각각에 대해 따로따로 돌려서 예측 결과 따로 따로 받는 것이 더 좋을 거라는 조언을 얻음. <br>



# 220722
jupyter lab에서 LGBMRegressor 돌릴 때 파쳐 네임에 한글 있으면 안됨. <br>
XGBRegressor 돌릴 때는 X_test까지도 feature네임에 한글 있으면 안됨. <br>

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


# 220726
## [앙상블과 cv?]
[앙상블 과제하면서 알아낸 것 cv, grid_search](https://velog.io/@ann9902/%EC%95%99%EC%83%81%EB%B8%94) <br>

- scaling <br>
tree기반 모델을 사용할 때는 scaling을 하지 않는다. <br>
그리고 scaling은 회귀에서도 하지 않는다는걸 어디에선가 들음. <br>
그러나 데이터 간의 거리가 매우 중요한 knn 같은 모델은 scaling에 민감. <br>

- one hot encoding <br>
`one hot encoding`은 범주 개수 만큼 피쳐를 생성하는 방법... -> 분류할 때 쓰는.. <br>

- 변수 선택 <br>
새로운 변수를 만드는 것이 아니라 원래 있는 변수 중에서 중요한 것만 선택하는 것으로 <br>
sklearn의 `SelectKBest`를 이욯하면 편하다. <br>

[cross validation을 하는 이유](https://velog.io/@skkumin/%EA%B5%90%EC%B0%A8-%EA%B2%80%EC%A6%9DCross-Validation) <br>

고정된 test set을 가지고 모델의 성능을 확인하고 파라미터를 수정하는 과정을 반복하면 결국 고정된 test data에 overfitting 발생 <br>
이러한 문제를 해결하기 위해 cross_validation을 이용. <br>
모든데이터가 쓰이기 때문에 데이터의 수가 적을때도 사용. <br>


[cv 방법들 설명 1](https://bluenoa.tistory.com/55) <br>
[2](https://kimdingko-world.tistory.com/167) <br>
[cross_val_scroe 사용법, 교차 검증을 간단하게](https://coding-potato.tistory.com/m/15) <br>
** estimator가 classifier 종류이면 내부적으로 stratified KFold 로 진행 <br>
[cv](https://aimb.tistory.com/138) 
모델이 새로운 데이터에 대해 어떻게 수행하는지 평가할 수 있는 기회를 잃게 된다. 이를 "data leakage"
[neg_mean_squared_error와 cv](https://thebook.io/007017/part02/ch05/05/01-03/)<br>



## [[앙상블 모형 이론 설명 good](https://velog.io/@changhtun1/ensemble)] <br>


[LGBM 성능 단번에 높이기](https://coding-potato.tistory.com/m/16) <br>

* get_clf_eval : 모델 평가 <br>
```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def get_clf_eval(y_test, pred=None, pred_proba=None):
    print('오차행렬 \n', confusion_matrix(y_test, pred))
    print('정확도 :', accuracy_score(y_test, pred))
    print('정밀도 : ',precision_score(y_test, pred))
    print('재현율 :', recall_score(y_test, pred))
    print('f1 score :', f1_score(y_test, pred))
    print('roc auc score :', roc_auc_score(y_test, pred_proba))

# 레이블 값이 극도로 불균형한 분포인 데이터의 경우 boost_from_average = True 값은 재현율과 roc_auc_score를 매우 저하시키므로
# 분류 지도학습 시 레이블 값이 극도로 불균형한 분포를 띄는 경우 LGBMClassifier의 파라미터 중 boost_from_average=False로 설정해주어야 함.
```
[스태킹 앙상블](https://hwi-doc.tistory.com/entry/%EC%8A%A4%ED%83%9C%ED%82%B9Stacking-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC) <br>
스태킹: 여러 가지 모델들의 예측값을 최종 모델의 학습 데이터로 사용하는 예측하는 방법 <br>

## [사용할 방법 정리]
1. 일단 데이터에 대한 정리
- 정형
- multi_output (target이 무려 14개)
- 비식별 data, 그러나 data의 각 feature에 대한 정보는 제공

2. 문제 정의
- 회귀
- 지도학습

3. 사용한 방법 정리
- `y 한 번에 예측`
    - multioutputregressor를 사용해봄 (성능이 그렇게 좋게 나오지는 않았음.)
- `y 각각 예측`
    - 피쳐 새로 생성 후 새로 만든 것만 사용 (만드는데 사용했다면 지움.) (feature engineering)
    - 피쳐 새로 생성 후 새로 만든 것도 사용 (기존 data 함께 사용.) (feature engineering)
        - y 각각에 대해 서로 상관이 있다면 하나 예측 후 다음 꺼 예측 시 feature로 사용하여 진행
        - x 중에 값이 계속 같은게 있다면 학습에 사용하지 않는다. (위와 동시에 진행)
        - 특정 y 하나에 대한 모델을 만들고 valid로 검증을 하고, 예측하는걸 하나의 프로세스로 진행


-----
- train_test_split의 인자에 대해 <br>
X, y, test_size = 0.8, shuffle = True <br>
test_size : Train에 대한 test 혹은 valid의 값의 개수의 비율 <br>
shuffle : train을 섞어서 나눌지 그냥 나눌지 <br>

** train data set과 test data set이 애초에 과거 data 대 최근 data 이런 식이라면 validation data set을 만들 때 shuffle을 하지 않기로 한다. <br>
<br>

## [grid search로 하이퍼파라미터 튜닝 후 각종 확인] <br>
```python
gs.cv_results_.keys()
gs.best_estimator_
f'최고 점수: {gs.best_score_}'
f'최적 하이퍼 파라미터: {gs.best_params_}'
model.get_params() # 내가 설정하지 않는 hp에대해서까지도 보여줌. 쓸데없이 길다.
model.get_params # 하면 내가 설정한 hyperparams 범위에 대해서만 어떤 걸로 돌아갔는지를 보여줌.

# best_params_는 gs에서만 쓸 수 있음. 개별 모델에 대해서는 쓸 수 없는 모듈임.

```


## 과제 생성 및 ...
가장 기본 ensemble: 제일 잘 나온 것의 예측값으로 평균 <br>

- 과제 (to 220728 목)
양재 AI hub로 데이터 셋으로 대회 제작 및 baseline 진행 <br>
data : train, test로 나누기 <br>

[read_excel 및 시트 지정 가능, 예제 코드](https://velog.io/@inhwa1025/Python-pandas%EB%A1%9C-exel-%ED%8C%8C%EC%9D%BC-%EC%9D%BD%EA%B8%B0) <br>
<br>

# 220728
- df의 유니크한 멀티인덱스 받는 코드(multi index) <br>
```python
list(data[['호선', '역번호', '역명']].value_counts().index.sort_values('호선')[0].unique())
```

- append와 extend의 차이 <br>
```python
li1 = [2]
li2 = [1,2]
li3 = [2,[3,4]]
li1.append(li2)
li1.append(li3) # inplace = True
li1
# 결과 : [2, [1, 2], [2, [3, 4]]]


li1 = [2]
li2 = [1,2]
li3 = [2,[3,4]]
li1.extend(li2)
li1.extend(li3) # inplace = True
li1
# 결과 : [2, 1, 2, 2, [3, 4]]


li1 = [2]
li2 = [1,2]
li3 = [2,[3,4]]
[li1, li2, li3] # inplace = False
# 결과 : [[2], [1, 2], [2, [3, 4]]]


li1 = [2]
li2 = [1,2]
li3 = [2,[3,4]]
li1+li2+li3 # inplace = False
# 결과 : [2, 1, 2, 2, [3, 4]]

```
<br>


- df의 틀을 다 짜놓고 값을 채워나가는 것은 사실상 불가능. <br>
한 row씩 작성해서 이어붙이는 방법밖에 없음. <br>
그것보다는 list로 한 번에 만들어서 한 번에 df화 시키는게 더 빠름 <br>
[ref 빈 df 만들어서 fot문으로 row 채우기](https://shydev.tistory.com/29) <br>
[하지 말아야 할 것들](http://daplus.net/python-%EB%B9%88-pandas-dataframe%EC%9D%84-%EB%A7%8C%EB%93%A0-%EB%8B%A4%EC%9D%8C-%EC%B1%84%EC%9A%B0%EC%8B%9C%EA%B2%A0%EC%8A%B5%EB%8B%88%EA%B9%8C/) <br>
```python
# df 생성 test cell

import random
# 빈 DataFrame 생성하기
df = pd.DataFrame(columns=['idx', 'number'])
for idx in range(1, 11):
    # 1과 100  사이의 random 한 값 생성하기
    number = random.randint(1, 101)
    # DataFrame에 특정 정보를 이용하여 data 채우기
    df = df.append(pd.DataFrame([[idx, number]], columns=['idx', 'number']), ignore_index=True)
    df
    break
    '#--------------------------'
df.set_index('idx', inplace=True)
df
```
<br>

[.to_dataframe()](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_dataframe.html) <br>
```python
import numpy as np
import pandas as pd
(np.array).to_dataframe
```
<br>


# dataset 생성 flow

0. 사용할 모든 df의 column 등의 형태와 dtype들을 통일시켜준다.
1. id가 될 수 있는 columns의 value들로 리스트를 만든다
    - `날짜` 리스트 (`datetime` 모듈을 이용해 2021-07-01 ~ 2022-05-31까지 빈날짜 없이.)
    - `호선, 역번호, 역명` 리스트
    - `구분` 리스트
    - `승객유형` 리스트

2. 만든 리스트로 조합을 만든다. (`itertools` 모듈의 `product` 함수를 이용한다.)
    - [itertools](https://ourcstory.tistory.com/414) <br>
        - `product` : 2개 이상의 리스트의 value들의 조합 (리스트 별로 꺼내서 조합을 만듦.)
        - `permutations` : 1개 이상의 리스트의 value들의 조합 (리스트들을 하나로 합쳐서 아무 value 랜덤으로 n개 뽑아서 조합을 만듦.)
        - `combinations` : permu와 동일, 그러나 순서가 바뀌어도 구성이 같으면 같은 조합으로 인식하여 추가 생성하지 않음.

3. 조합은 [mainid]가 된다. 조합의 개수만큼 row가 나와야한다.
4. 기존에 만들어둔 data df와 노인 df, 그리고 아직 만들지 않은 승객유형별21 df, 승객유형별22 df를 합쳐서 `alldf.csv`로 저장한다.
5. alldf.csv를 readlines로 불러와 이중리스트 형태로 `alllines` 변수에 저장한다.
6. alldf.csv를 readlines로 불러와 df의 id가 되는 부분을 슬라이싱하여 이중리스트 형태로 `isinid` 변수에 저장한다. (이 때 `alllines`와 `isinid`는 동일한 forloop에서 돌아가기 때문에 순서가 같으므로 index를 사용할 수 있다.)


7. `finlst`라는 이름의 공리스트를 만들고
8. [mainid]의 길이만큼 forloop을 돌리면서 `isinid`를 사용해 isinid에 있는지 여부 검사를 통해 <br>
    a. 있다면, 해당 loop에서의 mainid의 `isinid`에서의 위치로 `alllines`의 값을 인덱싱하여 <br>
    b. 없다면, 해당 loop에서의 mainid + [0]\*len(시간대)를 하여

9. `finlst`에 하나씩 삽입한다.

- -------------------- 여기까지 하고 저장하면 시간대가 승객별로 구분되지 않은 df가 나옴.

10. 일단 시간대를 리스트로 만들어 저장해놓은 변수 `col4findf`를 `finlst` 앞에 더하고 with open문을 사용하여 forloop으로 각 list를 str형으로 만들어 `findf.csv`로 <b>저장</b>한다.

11. target열을 만들어야 하므로 시간대가 승객별로 구분되지 않은 df인 `findf.csv`를 건들어서 승객유형별 시간대를 옆으로 따로 빼도록 한다.
    - 다음은 승객유형이 노인일 때의 시간대를 뽑는 예시 코드이다. <br>
    이렇게 뽑은 각 승객 유형별 df를 각각의 변수에 저장해놨다가 한 번에 옆으로 붙여주도록 한다. <br>

        ```python
        fin = pd.read_csv('findf.csv')
        시간대 = ['06시이전', '06시-07시', '07시-08시',
        '08시-09시', '09시-10시', '10시-11시', '11시-12시', '12시-13시', '13시-14시',
        '14시-15시', '15시-16시', '16시-17시', '17시-18시', '18시-19시', '19시-20시',
        '20시-21시', '21시-22시', '22시-23시', '23시이후']
        tp = list(fin.승객유형.unique()) # 여기에서 총은 빼고, 진행하고, 나중에 총까지의 열 뒤에 나머지를 이어붙여주도록 한다.
        del(tp[tp.index('총')])


        for i in tp: # loop는 유형 별로 돌아간다.
            globals()[f'{i}시간대df'] = fin[fin['승객유형']==i][시간대]

        ```
<br>


```python
li = ['l','d','s','a']
','.join(li)

# 결과 : 'l,d,s,a'
```
<br>

### 리스트보다 array가 더 빠르다.(고 하심.)


[.isocalendar()](https://codechacha.com/ko/python-how-to-get-which-weeks/) <br>
[망할 파이썬 문자열](https://www.delftstack.com/ko/howto/python/how-to-convert-string-to-datetime/) <br>
<br>


# 220729
[파이썬 튜플이 value인 리스트 정렬](https://hansuho113.tistory.com/28) <br>
[파이썬 딕셔너리 합치기](https://aplab.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%94%95%EC%85%94%EB%84%88%EB%A6%AC-%ED%95%A9%EC%B9%98%EA%B8%B0-%EC%82%AC%EC%A0%84-%EB%B3%91%ED%95%A9-%EB%B0%A9%EB%B2%95) <br>
```python
dic3 = {**dic1, **dic2}
```
<br>

[df drop](https://jimmy-ai.tistory.com/92) <br>
```python
# 열
train.drop('역명', axis = 1, inplace = True)
test.drop('역명', axis = 1, inplace = True)

# 행
경기인천 = list(np.where(data.자치구=='부평구')[0]) + list(np.where(data.자치구=='부천시')[0])
data.drop(경기인천, axis = 0, inplace = True)
```
<br>

[통계청 인구이동데이터 이동 경계 기준](https://kostat.go.kr/understand/info/info_qst/2/4/index.board?bmode=read&aSeq=161803) <br>
<br>

# 220801
[데이콘 카메라 이미지 품질 향상 AI 경진대회 baseline 코드](https://dacon.io/competitions/official/235746/codeshare/2874?page=2&dtype=recent) - 3등 기민님, 속도 개선 코드 <br>

[CNN 용어들](https://jetsonaicar.tistory.com/48) <br>
- 필터(filter) == 커널(kernel) == 가중치(weight)
- 채널(channel) : color
- stride(스트라이드), padding(패딩) : 커널 이동 보폭, output size 보존을 위한 방법
- activation map(액티베이션 맵), feature map(피쳐 맵) : Conv의 결과 레이어, training이 끝난 후
- pooling(풀링) 레이어 : 사이즈 다운

[Unet](https://velog.io/@guide333/U-Net-%EC%A0%95%EB%A6%AC) <br>
[backbone, head, neck](https://ddangjiwon.tistory.com/103) <br>
backbone : image to feature map <br>
neck : feature map 정제, 재구성 <br>
head : feature map의 location 작업 수행<br>
[백본과 알고리즘의 차이 추가 설명](https://www.inflearn.com/questions/34244) <br>
- Q. 계속 나오는 단어들에 대한 의문 <br>
    강의를 들으면서 알고리즘과 Backbone이 서로 어떻게 다른건지 잘 구분이 되지 않습니다.

    StackOverflow에서 찾아보니 Backbone을 아래와 같이 묘사하는 것을 확인했습니다.
    -  Feature Extractor
    -  Feature Extracting Network
    - Neural Network Architecture
 
    그렇지만 여전히 알고리즘과 Backbone이 정확하게 어떻게 다른건지 잘 구분이 가지 않는 상황입니다.

- A.
Feature Extractor는 일반적으로 저희가 익숙한 CNN Classification에 사용되는 네트웍 모델에서 최종 classification layer만 제거된 모델을 이용. <br>
주요 CNN Classification 네트웍 모델인 VGGNET, INCEPTION, RESNET 등을 Feature Extractor로 이용하며, 이들 네트웍 모델에서 Fully connected Layer 부분을 제거 <br>
원본 이미지를 feature map으로 변환하는 layer만 Object Detection 모델에서 사용 <br>

- object detection : 객체 탐지 <br>

[computer vision 방법론 - 데이콘 카메라 이미지 품질 향상 AI 경진대회 baseline 코드 review](https://tistory-nari.tistory.com/26) <br>

[regression 모델 평가 방법](https://brunch.co.kr/@chris-song/34) <br>

psnr_score : 최대 신호 대 잡음 비 <br>
mse가 작을 수록 더 큰 psnr을 갖게 됨.
[papers with code Image Super Resolution](https://paperswithcode.com/task/image-super-resolution) <br>
[SRGAN 논문 정리](https://dydeeplearning.tistory.com/2) <br>


# 220802-3
`excel to csv.ipynb` <br>

sheet_name = number, df mapping 등... <br>
<br>

# 220803
`디렉토리 내 파일 자동호출.ipynb` <br>
glob과 os 라이브러리 <br>
<br>


\는 에러가 남. 아래는 예시<br>
```python
'd\d\f'
```
```
<출력결과>
'd\\d\x0c'
```
<br>

```python
path = 'C:\Users\uuuu\Downloads\'
```
```
<출력 에러>
(unicode error) 'unicodeescape' codec can't decode bytes in position 2-3: truncated \UXXXXXXXX escape

또는
EOL while scanning string literal
```
<br>
<br>

list mapping, lambda 내 if 문 등... <br>

df랑 list는 mapping 방식이 다름.

```python
## df의 경우
df[열].map(lambda x : 인자 x를 포함한 함수)

# 또는
df[열].map({'기존': '바꾸고싶은 내용', ...})


## list의 경우
list(map(lambda x : 인자 x를 포함한 함수 if ... else ..., list_name))
# if 문을 쓸거면 else는 꼭 써줘야 함. 단, pass, continue가 아닌 None 또는 그 자체를 반환하는 식으로
```

```python
## 아래 함수를 작성했을 때, 함수 아래 <4개 식과 for문>은 모두 같은 값을 반환함.
def index_p(x):
    if '.' in x:
        return x.index('.')

list(map(index_p, file_list))
list(map(lambda x: index_p(x), file_list))
list(map(lambda x: x.index('.') if '.' in x else print('None'), file_list))
list(map(lambda x: x.index('.') if '.' in x else None, file_list))

for i in file_list:
    if '.' in i:
        i.index('.')
# [0, None, None, None, None, None, 14, 14, 11, 19, 15, 33, 16, 23]
```


[os](https://bskyvision.com/762) <br>
[glob](https://wikidocs.net/83) <br>
[리스트 내 문자 포함 확인](https://ddolcat.tistory.com/676) <br>
[map함수를 사용하는 것과 아닌 것의 차이](https://blockdmask.tistory.com/531) <br>
[lambda 표현식에 조건부 표현식 사용하기](https://dojang.io/mod/page/view.php?id=2360) <br>
<br>

[python re library, 정규표현식 regular expression](https://jjuha-dev.tistory.com/entry/Python-%EC%A0%95%EA%B7%9C%ED%91%9C%ED%98%84%EC%8B%9D-resub%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-%EB%AC%B8%EC%9E%90%EC%97%B4-%EC%B9%98%ED%99%98%ED%95%98%EA%B8%B0) <br>
[numpy concatenate, 그냥 axis대로 이어붙이기](https://engineer-mole.tistory.com/234) <br>

- `np.where` 사용법
```python
# g_data가 list인 경우에 ndarray로 바꿔서 진행 가능
np.where(np.array(g_data)=='문자열')

# g_data가 ndarray인 경우에 바로 가능
np.where(g_data == 'V2000')
```

- ndarray또는 df의 특정 열에 대해 값이 있는지 여부를 확인하는 `.isin()` 사용법
일단 이건 아마도 DSML readme에 자세히 나와있을 것임. (앗, 없음...;;; 어디에도 없음.) <br>
괄호 안에 문자열이든 숫자든 아무튼 `찾고 싶은 값`을 그냥 넣을 경우 list로 안받아졌다고 에러가 난다. <br>
```python
tmp_g[tmp_g.isin(['V2000'])]
tmp_g[tmp_g.isin([값1, 값2])]
```
<br>

```python
## 혹시 찾은 그 값이 들어있는 행을 지우고 싶다면
df.drop(df[df['Mary'].isin([68, 70])].index, inplace = True) # df의 Mary 열에 68, 70이 있는 행은 다 지운다.
df_del = df.reset_index(drop = True)


### 이하의 한 줄은 위의 drop의 일련의 과정과 같은 결과를 낸다.
df_del = df.drop(df[df['Mary'].isin([68, 70])].index).reset_index(drop = True)
```
👆👆👆이런 식으로 사용하도록 하자. <br>
<br>


- 그림이 아니어도 CNN을 사용할 수 있다. <br>
- 💛💛💛*** 다시 말하지만 모델 층 쌓는 게 딥러닝이다. <br>
<br>


# 220804 ~ 220805
API Crawling, code updated <br>
HTTP 04 error 발생 문제 -> 월요일에 운영계정으로도 에러나면 전화로 해결 <br>


# 220811
.isin()을 통해 a.df에는 없는 b.df의 row만을 뽑으려면 <br>
column을 특정해서 설정해줘야한다. <br>
그러지 않으면 .isin()의 결과가 False인 것도 출력이 됨. <br>
```python
b_df[b_df['cid'].isin(a_df['cid'])] # isin은 column을 특정해줘야함.
```
<br>

그리고 .isin()은 괄호 안에 값(들어있는지 확인하고 싶은 값)을 list나 array 형태로 넣어줘야함. <br>
```python
ex[ex.cid.isin(['1957446'])]
```
<br>

내가 지금 하고자하는 작업은 df를 최종적으로 합쳤을 때 <br>
cids.csv 안에 example_df.csv의 cid중 ___안들어있는게 뭔지___ 알고싶은 것임. <br>

[작업 코드] <br>
```python
str_cid = [str(i) for i in tmp.cid] # tmp의 cid는 int, CIDs의 tmp는 str이라 형변환을 해줘야 검색이 가능함.
CIDs[~CIDs.cid.isin(str_cid)]
```
<br>


# 220921
qr코드 생성을 위한 동적 크롤링 (서버에서 동적 크롤링 안되는게 아니었음. chromedriver가 문제였던 것임.) <br>
chromedriver 파일 위치 찾음. (파일 위치 찾는 데에는 아래의 코드 이용, 모든 위치 반환, 그냥 상단에서 부터 하나씩 실행시켜봄. 위치는 dacon/Dacon/HDD_01/a.outmember/~~~였음.) <br>

``` bash
find -name 'chromedriver'
```
<br>

[각종 selenium browser 열 때 설정, 창(window) 사이즈 등](https://incomeplus.tistory.com/266) <br>

```python
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('diisable-gpu')

driver = webdriver.Chrome('chromedriver', chrome_options = options)
```
<br>

[브라우저 띄우지 않은 채로 크롤링 하기 (서버에서 실행하면 어차피 안뜸), 이거 설정하면 로컬에서는 일단 에러나서 사용 안하는 편이 좋겠음.](https://goodthings4me.tistory.com/196) <br>

```python
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless') # 브라우저 안뜨게
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')

## webdriverChrome()할 때 options parameter에 chrome_options 변수 넣어주기
driver = webdriver.Chrome('chromedriver', options = chrome_options)
```
<br>

[.click()이 안먹힐 때](https://wkdtjsgur100.github.io/selenium-does-not-work-to-click/) <br>

```python
# 다음과 같이 시도해 보았다.
driver.find_element_by_xpath("//form[@class='ui form']/button").click()
## 아무리 해도 클릭이 되지 않음.

## 그래서 is_enabled()를 사용해서 해당 엘리먼트가 클릭 가능한 지 테스트 해봤지만 True를 반환
driver.find_element_by_xpath("//form[@class='ui form']/button").is_enabled()

## 다음과 같이 해결
from selenium.webdriver.common.keys import Keys
driver.find_element_by_xpath("//form[@class='ui form']/button").send_keys(Keys.ENTER)


## 만약 그래도 안된다면, 다음과 같은 명령문 사용
element = driver.find_element_by_xpath("//form[@class='ui form']/button")
driver.execute_script("arguments[0].click();", element)
```
<br>

[webdriver 공식 doc](https://w3c.github.io/webdriver/#element-send-keys) <br>

[selenium 공식 doc](https://www.selenium.dev/documentation/webdriver/elements/interactions/) <br>
<br>

# 220922
- gridsearch에서 점수에 따른 모델 찾기

```python
gscv.cv_results_.keys()
gscv.cv_results_['rank_test_score']
gscv.cv_results_['mean_test_score']
gscv.cv_results_['params']
```
<br>

- markdown 수식 <br>
$ X+Y=Z; $
<br>

[markdown](https://velog.io/@d2h10s/LaTex-Markdown-%EC%88%98%EC%8B%9D-%EC%9E%91%EC%84%B1%EB%B2%95)
<br>
[markdown 하이라이팅 지원 언어](https://github.com/jincheng9/markdown_supported_languages)
<br>

# 220929, 목 - 보조강사 출장, 제주 테크노파크
[folium 사용법 강의 - 송우석님](https://github.com/GaYounKoh/work/blob/main/%EC%A0%9C%EC%A3%BC%20%ED%85%8C%ED%81%AC%EB%85%B8%ED%8C%8C%ED%81%AC/220929%2C%20%EC%9A%B0%EC%84%9D%EB%8B%98%20%EA%B0%95%EC%9D%98%20folium.ipynb) <br>
[plotly dash 사용 예시](https://dash.gallery/Portal/?_gl=1*1lqfull*_ga*NTE1OTMzMzI2LjE2NjQ0MTA3NTY.*_ga_6G7EE0JNSC*MTY2NDQyOTQzMi4yLjEuMTY2NDQyOTQzNC4wLjAuMA..) <br>


## 221006, 목
[크롤링 에러 - 아마도 크롬 드라이버 버전이슈가 아닐까..., DeprecationWarning: executable_path has been deprecated, please pass in a Service object](https://velog.io/@kite_day/selenium-%EC%98%A4%EB%A5%98-%ED%95%B4%EA%B2%B0-%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%9C%BC%EB%A1%9C-%ED%81%AC%EB%A1%A4%EB%A7%81-%EC%98%A4%EB%A5%98) <br>



[webdriver-manager official doc](https://pypi.org/project/webdriver-manager/) <br>
[크롬 드라이버 실시간으로 설치하여 사용하기](https://liveyourit.tistory.com/14) <br>
-> <b>웹 드라이버매니저 이용</b> <br>
- <b>버전 변경에 상관없이 현재 OS에 설치된 크롬브라우저를 사용합니다.</b> <br>
[크롤링 소소한 팁](https://pythonblog.co.kr/coding/7/) <br>
[크롬 웹드라이버 사용해야하는 이유와 유의사항, 전체 코드](https://redfox.tistory.com/m/87) <br>
- 유의사항 <br>
기존 방식의 경우 웹드라이버 지정 과정에서만 브라우저 유형을 지정하는게 전부였지만, 웹드라이버 매니저의 경우 모듈 import(임포트) 과정에서부터 브라우저 유형을 지정해줘야해서 다소 귀찮아질 수 있다. <br>

[크롬 웹드라이버 에러 관련 팁](https://www.codeit.kr/community/threads/22660) <br>
```
앞서 다른 분이 크롬 웹드라이버 임포트 할 때의 에러를 공유해 주신 것과 관련해서 저도 팁 하나 공유 드립니다.
코드잇 측에서 포스팅 해주신 코드에서는 크롬 드라이버를 직접 다운 받아서 실행 시키는 정석적인 방법인데,
(1) 드라이버 파일 경로를 잘못 지정하거나, (2) 드라이버 버전이 맞지 않아서 에러가 발생 할 수 있습니다. ( 저는 후자였습니다.)

드라이버 재설치 혹은 경로 지정에 어려움을 겪는 경우, 아래와 같은 코드가 도움이 될 수 있어서 공유 드립니다. 아래와 같이 패키지 설치 하시고, 구문을 추가/대체 하시면 무난히 진행됩니다. ([출처](https://stackoverflow.com/questions/60296873/sessionnotcreatedexception-message-session-not-created-this-version-of-chrome))
```
<br>

[최종 에러 Chrome failed to start: crashed 해결](https://league-cat.tistory.com/278) <br>


## 221011
인증서 발급 자동화 프로세스 (크롤링 + pdf 위에 폰트)<br>
EDA 프로세스 <br>

[TOC 생성 코드](https://gmnam.tistory.com/246) <br>
``` markdown
핵심 프로세스
- header text옆에 a tag 추가
- class는 anchor로 고정
- TOC 생성에 이용할 유니크한 id 생성

아래는 예시
<a class="anchor" id="chapter1"></a>
```

<br>

## 221012
spline smoothing (스플라인 스무딩)
데이터에 대해 곡선을 적합시키는 작업 https://cdm98.tistory.com/27 <br>

동일한 정보를 제공해준다면 더욱 간단한 모델을 사용하는 것이 더욱 효율적 <br>

자유도가 낮을 수록 간단한 모델인 것? <br>


## 221107
null인 값이 들어있는 열 찾기
```python
idx = np.where(df.isnull().sum())
df.columns[idx]
```

null인 값이 들어있는 행 찾기 (특정 열 줘야 할 수 있음., 위와 이어지는 코드임.)
```python
np.where(df[df.columns[idx]].isnull())
```

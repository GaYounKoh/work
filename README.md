# work
internship

# 220719
0. 인사!!!
~~1. mac에서 vscode하는 방법 찾아야 함. (그 전에 os에 대해 고민 해봐야함. mac 너무 어려움. mac은 뭐라도 익히고 난 다음에..., 아주 기본적인 단축키 조차 못써서 답답함.)~~ 포기하고 일단 내 노트북 쓰는중.
2. python으로 SQL하는 코드 익히기 (불러오는 코드 정도만 익히면 판다스는 문제 없음.(? 아마도...? 설마..?))
~~3. ip 회사 서버에 연결~~ 연결 완료!, 집에서만 쓸 수 있음
~~4. 회사 바로 앞 아샷추의 위치를 파악해봐야함. (일단 이디야는 없음.)~~ 아이스티 사다놓음.
5. 점심시간 1시간, 12:30 ~ 13:30 ☆☆☆, 샌드위치집 위치 알아두기
6. 대회에 대해 익히기
7. AImers는 빨리 전화해봐야할듯.
~~8. 금요일 빨리 집~~ 일단 8시 전까지면 됨.
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


# 220726
## 앙상블과 cv?
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



## [앙상블 모형 이론 설명 good](https://velog.io/@changhtun1/ensemble) <br>


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
<br>
[스태킹 앙상블](https://hwi-doc.tistory.com/entry/%EC%8A%A4%ED%83%9C%ED%82%B9Stacking-%EC%99%84%EB%B2%BD-%EC%A0%95%EB%A6%AC) <br>
스태킹: 여러 가지 모델들의 예측값을 최종 모델의 학습 데이터로 사용하는 예측하는 방법 <br>

## 사용할 방법 정리
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
test_size : Train에 대한 test 혹은 valid의 값의 개수의 비율
shuffle : train을 섞어서 나눌지 그냥 나눌지

** train data set과 test data set이 애초에 과거 data 대 최근 data 이런 식이라면 validation data set을 만들 때 shuffle을 하지 않기로 한다.




grid search로 하이퍼파라미터 튜닝 후 각종 확인
```python
gs.cv_results_.keys()
gs.best_estimator_
f'최고 점수: {gs.best_score_}'
f'최적 하이퍼 파라미터: {gs.best_params_}'
model.get_params() # 내가 설정하지 않는 hp에대해서까지도 보여줌. 쓸데없이 길다.
model.get_params # 하면 내가 설정한 hyperparams 범위에 대해서만 어떤 걸로 돌아갔는지를 보여줌.

# best_params_는 gs에서만 쓸 수 있음. 개별 모델에 대해서는 쓸 수 없는 모듈임.

```

```python
import sklearn
sorted(sklearn.metrics.SCORERS.keys())
# 평가방식들 보는 코드
```


회사 주피터 랩 n_jobs = 16정도로 진행할 것. 내부 cpu 개수에 따라 조정해놨던 것



## 과제 생성 및 ...
가장 기본 ensemble: 제일 잘 나온 것의 예측값으로 평균

- 과제
양재 AI hub로 데이터 셋으로 대회 제작 및 baseline 진행 <br>
data : train, test로 나누기 <br>

[read_excel 및 시트 지정 가능, 예제 코드](https://velog.io/@inhwa1025/Python-pandas%EB%A1%9C-exel-%ED%8C%8C%EC%9D%BC-%EC%9D%BD%EA%B8%B0)
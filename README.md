# work
internship

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



### error message
```
Expected 2D array, got 1D array instead:
array=[val val val val val val val].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```
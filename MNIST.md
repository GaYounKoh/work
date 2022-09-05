1. Library Load
2. 이미지 추출(array), 출력(그림), 라벨list 생성 함수
3. 환경설정 (GPU, 하이퍼파라미터, random seed 고정)
4. 사용자 정의 Dataset 클래스 -> 이미지 arr는 transform 적용 후 flatten, label 함께 return
5. transform 설정 -> ToTensor()
6. train . val split
7. DataLoader # batch사이즈 만큼씩 데이터 묶어주는 함수 (pytorch에서 import)
8. Model generate
9. Train
10. Test
11. 답지 Load
12. Metric

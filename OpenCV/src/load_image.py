import cv2

# 지정한 파일을 컬러 포맷으로 불러온다.
# IMREAD_COLOR는 컬러 포맷으로 영상을 읽겠다는 의미이다.
# 이미지를 img_color 변수에 넘파이 배열로 대입된다.
img_color = cv2.imread("cat on laptop.png", cv2.IMREAD_COLOR)

# 이미지를 불러올 수 없다면 img_color 변수는 None
# 이미지를 불러올 수 없는 경우를 체크
if img_color is None:
    print("이미지 파일을 읽을 수 없습니다.")
    exit(1)

# 이미지를 보여줄 윈도우를 생성
# 지정한 문자열이 윈도우의 타이틀바에 보이게 됨.
cv2.namedWindow('Color')

# 윈도우 식별자가 "Color"인 윈도우에 변수 img_color가 가리키는 넘파이 배열에 저장된 이미지를 보여줌.
# imshow만 사용해도 이미지를 보여줌.
cv2.imshow('Color', img_color)

# ms 단위로 지정한 시간만큼 대기함.
# 0인 경우 => OpenCV로 생성한 윈도우 창이 선택된 상태에서 키보드 입력이 있을 때까지 대기
cv2.waitKey(0)

# 사용이 끝난 윈도우를 종료해줌.
cv2.destroyAllWindows()
# 읽어온 이미지를 그레이 스케일 이미지로 변환 후 저장하기

import cv2

# 컬러 이미지로 파일을 불러옴.
img_color = cv2.imread("cat on laptop.png", cv2.IMREAD_COLOR)

if img_color is None:
    print("이미지 파일을 읽을 수 없습니다.")
    exit(1)

# 컬러 이미지를 먼저 화면에 보여주고
cv2.namedWindow('Color')
cv2.imshow('Color', img_color)

# 대기하다가 키보드 입력이 있으면 다음 줄이 실행됨.
cv2.waitKey(0)

# img_color에 저장된 컬러 이미지를 그레이 스케일로 변환 후 img_gray에 대입.
# COLOR_BGR2GRAY는 BGR 컬러 이미지를 그레이 스케일로 변환함.
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# namedWindow 함수는 생략 가능함.
# img_gray에 저장된 그레이 스케일 이미지를 식별자가 "Grayscale"인 창에 보여줌.

# 컬러 이미지를 보여줄 때 사용한 "Color"를 사용하도록 수정하면 그레이 스케일 이미지가 "Color" 창에 보이게 됨.
cv2.imshow("Grayscale", img_gray)

# img_gray에 저장된 이미지를 grayscale.png로 저장함 => 이미지 포맷은 지정한 파일의 확장자에 따라 결정됨.
cv2.imwrite("grayscale.png", img_gray)

# 아무 키나 누르면 프로그램이 종료됨.
cv2.waitKey(0)
cv2.destroyAllWindows()
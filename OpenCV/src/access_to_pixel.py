# 픽셀에 접근하는 방법

import cv2 as cv
import numpy as np

img_color = cv.imread("apple.png", cv.IMREAD_COLOR)

# 이미지의 높이와 너비를 가져옴.
height, width = img_color.shape[:2]

# 그레이 스케일 이미지를 저장할 넘파이 배열을 생성함.
img_gray = np.zeros((height, width), np.uint8)

# for 문을 돌면서 (x, y)에 있는 픽셀을 하나씩 접근함.
for y in range(0, height):
    for x in range(0, width):

        # 컬러 이미지의 (x, y)에 있는 픽셀의 b, g, r 채널을 읽는다.
        b = img_color.item(y, x, 0)
        g = img_color.item(y, x, 1)
        r = img_color.item(y, x, 2)

        # (x, y) 위치의 픽셀에 그레이 스케일값이 저장된다.
        # 평균값을 사용하는 경우
        # gray = int((r + g + b) / 3.0)
        # BT.709에 명시된 비율 사용하는 경우
        gray = int(r * 0.2126 + g * 0.7152 + b * 0.0722)

        img_gray.itemset(y, x, gray)


# 결과 이미지에 컬러를 표시하기 위해 컬러 이미지로 변환함.
img_result = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)

# y의 범위가 150 ~ 200, x의 범위가 200 ~ 250인 영역의 픽셀을 초록색 픽셀로 변경함.
for y in range(150, 201):
    for x in range(200, 251):

        img_result.itemset(y, x, 0, 0)       # b
        img_result.itemset(y, x, 1, 255)     # g
        img_result.itemset(y, x, 2, 0)       # r

cv.imshow("color", img_color)
cv.imshow("result", img_result)

cv.waitKey(0)

cv.destroyAllWindows()
# 같은 이미지 데이터 공유

import cv2 as cv

img_gray = cv.imread('cat on laptop.png', cv.IMREAD_GRAYSCALE)

# 대입 연산자
img_copyed1 = img_gray

# img_gray와 img_copyed1은 같은 넘파이 배열을 가리키게 된다.
# 같은 객체이므로 id 함수의 리턴값이 같다. 
print(id(img_gray), id(img_copyed1))

# 아직은 똑같은 넘파이 배열을 가리키기 떄문에 
# img_gray에 선을 그리면 img_copyed1에서 선이 그려진다.
cv.line(img_gray, (0, 0), (100, 100), 0, 10)

# img_gray에 이진화를 적용하여 결과를 img_gray에 저장하면
# img_gray와 img_copyed1은 별개의 넘파이 배열을 가리키게 된다.
ret, img_gray = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY)

# 다른 객체가 되므로 id 함수의 리턴값이 달라진다.
print(id(img_gray), id(img_copyed1))

# img_copyed1에는 영향을 주지 못하고 img_gray만 이진화된다.
cv.imshow("img_gray", img_gray)
cv.imshow("img_copyed1", img_copyed1)

cv.waitKey(0)
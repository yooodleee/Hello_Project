import cv2 as cv

img_gray = cv.imread("cat on laptop.png", cv.IMREAD_GRAYSCALE)

# [start_y:end_y, start_x:end_y]로 ROI 영역을 지정함.
img_sub1 = img_gray[20 : 20 + 150, 20 : 20 + 150]

# img_color의 일부 데이터를 공유하기 때문에 True가 리턴된다.
print(img_sub1.base is img_gray)

# 같은 이미지 데이터를 공유하기 때문에 img_sub1에 선을 그리면
# img_gray에도 선이 그려진다.
cv.line(img_sub1, (0, 0), (100, 100), 0, 10)

ret, img_sub1 = cv.threshold(img_sub1, 127, 255, cv.THRESH_BINARY)

# img_sub1가 입력과는 다른 객체가 되므로 False가 리턴된다.
print(img_sub1.base is img_gray)

# img_color에는 영향을 주지 못하고 img_sub1만 이진화된다.
cv.imshow("img_gray", img_gray)
cv.imshow("img_sub1", img_sub1)

cv.waitKey(0)
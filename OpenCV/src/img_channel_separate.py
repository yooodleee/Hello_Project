import cv2 as cv

img_color = cv.imread("color.png", cv.IMREAD_COLOR)

# 컬러 이미지를 채널별로 분리함.
img_b, img_g, img_r = cv.split(img_color)

# 채널별 이미지를 조합하여 컬러 영상을 생성함.
# blue와 red의 순서를 바꾸었음.
img_result = cv.merge((img_r, img_g, img_b))

cv.imshow("Color", img_result)
cv.imshow("B", img_b)
cv.imshow("G", img_g)
cv.imshow("R", img_r)

cv.waitKey(0)
cv.destroyAllWindows()
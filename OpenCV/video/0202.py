# 영상 파일 저장

import cv2


imageFile = './data/lena.jpg'
img = cv2.imread(imageFile) # cv2.imread(imageFile, cv2.IMREAD_COLOR)

cv2.imwrite('./data/Lena/bmp', img)
cv2.imwrite('./data/Lena.png', img)
cv2.imwrite('./data/Lena2.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
cv2.imwrite('./data/Lena2.png', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
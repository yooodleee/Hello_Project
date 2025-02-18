# Matplotlib 1: 컬러영상 표시

import cv2
from matplotlib import pyplot as plt


imageFile = './data/lena.jpg'
imgBGR = cv2.imread(imageFile)  # cv2.IMREAD_COLOR
plt.axis('off')

# plt.imshow(imgBGR)
# plt.shw()

imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BRG2RGB)
plt.imshow(imgRGB)
plt.show()